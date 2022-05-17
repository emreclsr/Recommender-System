#############################################
# ASSOCIATION RULE BASED RECOMMENDER SYSTEM
#############################################

# İş Problemi
# 3 farklı kullanıcının sepetinde yer alan ürüne göre öneri yapılması
# Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si: 23235
# Kullanıcı 3’ün sepetinde bulunan ürünün id'si: 22747


# Veri Seti Hikayesi
# Online Retail II isimli veri seti bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki
# online satış işlemlerini içermektedir.
#
# Veri seti 8 değişkene sahiptir.
# InvoiceNo: Fatura numarası (Eğer bu kod C ile başşlıyorsa işlemin iptal edildiğini ifade eder.)
# StockCode: Ürün kodu (Her bir ürün için eşsiz)
# Description: Ürün ismi
# Quantity: Ürün adedi (Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate: Fatura tarihi
# Price: Birim fiyatı (Sterlin)
# CustomerID: eşsiz müşteri numarası
# Country: Ülke ismi

# İşlem Adımları
# 1. Veri setinin hazırlanması
# 2. Alman müşteriler üzerinden birliktelik kurulması
# 3. Sepet içerisindeki ürün Id’leri verilen kullanıcılara ürün önerisinde bulunma

#############################################
# Görev 1: Veriyi Hazırlama
#############################################

# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("week4/recommender_systems/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz.
# (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)

df = df[df["StockCode"] != "POST"]

# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.

df.isnull().sum()
df.dropna(inplace=True)

# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)

df = df[~df["Invoice"].str.contains("C", na=False)]

# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.

df = df[df["Price"] > 0]

# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.

df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T

"""
Aykırı değerler bulunmaktadır. 99% percentile'da değer Quantity için 120, Price için 16.95 bulunmaktadır.
Maksimum değerleri ise Quantity ve Price için sırasıyla, 80995 ve 4161.06'dır. Bu değerler aykırılık
teşkil ekmektedir.
"""


def outlier_thresholds(dataframe, variable, int_th=False, low_th=0.01, up_th=0.99):
    """
    Returns the lower and upper limits for outliers in a variable. If int_th is True, the thresholds return as int.

    Parameters
    ----------
    dataframe: pd.DataFrame
    variable: str
    int_th: bool, default False
    low_th: float, default 0.01
    up_th: float, default 0.99

    Returns
    -------
    low_limit: int or float
    up_limit: int or float

    """
    quartile1 = dataframe[variable].quantile(low_th)
    quartile3 = dataframe[variable].quantile(up_th)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    if int_th:
        return int(low_limit), int(up_limit)
    else:
        return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, int_th=False, low_th=0.01, up_th=0.99):
    """
    Replaces outliers in a variable with the threshold values. If int_th is True, the thresholds return as int.

    Parameters
    ----------
    dataframe: pd.DataFrame
    variable: str
    int_th: bool, default False
    low_th: float, default 0.01
    up_th: float, default 0.99

    Returns
    -------
    None

    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, int_th, low_th, up_th)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "Quantity", int_th=True)
replace_with_thresholds(df, "Price")

df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T

#######################################################################
# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
#######################################################################


# Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.


def create_invoice_product_df(dataframe, id=False):
    """
    Creates a pivot table with the product description and invoice no.
    Created pivot table shows that, each product included in invoice or not.

    Parameters
    ----------
    dataframe: pd.DataFrame
    id: bool, default False

    Returns
    -------
    dataframe: pd.DataFrame
    """
    if id:
        dataframe = dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0). \
                    applymap(lambda x: 1 if x > 0 else 0)
    else:
        dataframe = dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0). \
                    applymap(lambda x: 1 if x > 0 else 0)

    return dataframe

# Adım 2: Kuralları oluşturacak create_rulesfonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.


def create_rules(dataframe, country_name="Germany"):
    """
    Creates association rules for the given country.

    Parameters
    ----------
    dataframe: pd.DataFrame
    country_name: str, default "Germany"

    Returns
    dataframe: pd.DataFrame
    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame
    -------

    """
    dataframe = dataframe[dataframe["Country"] == country_name]
    dataframe = create_invoice_product_df(dataframe, id=True)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)  # çok fazla verinin kaybolmaması için %1 alıyoruz.
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

    return dataframe, frequent_itemsets, rules


pivot_german, freq_item_german, rules_german = create_rules(df, country_name="Germany")


#####################################################################################################
# Görev 3: Sepet İçerisindeki Ürün Id’leriVerilen Kullanıcılara Ürün Önerisinde Bulunma
#####################################################################################################

# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df, 21987)  # ['PACK OF 6 SKULL PAPER CUPS']
check_id(df, 23235)  # ['STORAGE TIN VINTAGE LEAF']
check_id(df, 22747)  # ["POPPY'S PLAYHOUSE BATHROOM"]

# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.


def arl_recommender(rules_df, product_id, rec_type="lift", rec_count=1):
    """
    Gives a recommended product or products for the given product id.

    Parameters
    ----------
    rules_df: pd.DataFrame
    product_id: int
    sort_type: "lift" or "confidence" or "support", default "lift"
    rec_count: int, default 1

    Returns
    -------
    recommendation_list: list
    """
    sorted_rules = rules_df.sort_values(rec_type, ascending=False)
    recommendation_list = []
    x = 0
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
                x += 1

                if x == rec_count:
                    return recommendation_list


rec_1 = arl_recommender(rules_german, 21987, "lift", 1)
rec_2 = arl_recommender(rules_german, 23235, "lift", 1)
rec_3 = arl_recommender(rules_german, 22747, "lift", 1)

# Adım 3: Önerilecek ürünlerin isimlerine bakınız.

prod_1 = check_id(df, rec_1[0])  # ['SET/6 RED SPOTTY PAPER CUPS']
prod_2 = check_id(df, rec_2[0])  # ['ROUND STORAGE TIN VINTAGE LEAF']
prod_3 = check_id(df, rec_3[0])  # ["POPPY'S PLAYHOUSE BEDROOM "]

