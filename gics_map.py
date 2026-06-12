#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一行业口径:GICS 一级(11 个大类)+ 二级(24 个行业组,REITs 并入房地产)。
三个市场的原生行业分类在抓取时映射到这套口径,每条记录写入:
  sec = 一级大类   g = 二级行业组
前端不做任何映射,直接用这两个字段。
"""

# 二级 -> 一级
GRP2SEC = {
    "能源": "能源",
    "原材料": "原材料",
    "资本品": "工业", "商业和专业服务": "工业", "运输": "工业",
    "汽车与汽车零部件": "可选消费", "耐用消费品与服装": "可选消费",
    "消费者服务": "可选消费", "可选消费零售": "可选消费",
    "日常消费零售": "日常消费", "食品饮料与烟草": "日常消费", "家庭与个人用品": "日常消费",
    "医疗保健设备与服务": "医疗保健", "制药与生物科技": "医疗保健",
    "银行": "金融", "金融服务": "金融", "保险": "金融",
    "软件与服务": "信息技术", "技术硬件与设备": "信息技术", "半导体与半导体设备": "信息技术",
    "电信服务": "通讯服务", "媒体与娱乐": "通讯服务",
    "公用事业": "公用事业",
    "房地产": "房地产",
}

# 各一级大类的缺省二级(细分行业映射结果与东财大类冲突时回退用)
SECTOR_DEFAULT_GRP = {
    "能源": "能源", "原材料": "原材料", "工业": "资本品",
    "可选消费": "消费者服务", "日常消费": "食品饮料与烟草",
    "医疗保健": "医疗保健设备与服务", "金融": "金融服务",
    "信息技术": "软件与服务", "通讯服务": "电信服务",
    "公用事业": "公用事业", "房地产": "房地产",
}
# 东财美股大类的两个别名归一
US_SECTOR_ALIAS = {"非日常生活消费品": "可选消费", "日常消费品": "日常消费"}

# A股(东财业绩报表"所处行业",申万二级口径)-> GICS 二级
A2GRP = {
    "IT服务Ⅱ": "软件与服务", "软件开发": "软件与服务",
    "计算机设备": "技术硬件与设备", "通信设备": "技术硬件与设备", "消费电子": "技术硬件与设备",
    "光学光电子": "技术硬件与设备", "元件": "技术硬件与设备", "其他电子Ⅱ": "技术硬件与设备",
    "半导体": "半导体与半导体设备",
    "通信服务": "电信服务",
    "游戏Ⅱ": "媒体与娱乐", "数字媒体": "媒体与娱乐", "广告营销": "媒体与娱乐",
    "影视院线": "媒体与娱乐", "出版": "媒体与娱乐", "电视广播Ⅱ": "媒体与娱乐",
    "中药Ⅱ": "制药与生物科技", "化学制药": "制药与生物科技", "生物制品": "制药与生物科技",
    "动物保健Ⅱ": "制药与生物科技",
    "医疗器械": "医疗保健设备与服务", "医疗服务": "医疗保健设备与服务",
    "医疗美容": "医疗保健设备与服务", "医药商业": "医疗保健设备与服务",
    "银行Ⅱ": "银行", "证券Ⅱ": "金融服务", "多元金融": "金融服务", "保险Ⅱ": "保险",
    "乘用车": "汽车与汽车零部件", "商用车": "汽车与汽车零部件",
    "汽车零部件": "汽车与汽车零部件", "汽车服务": "汽车与汽车零部件", "摩托车及其他": "汽车与汽车零部件",
    "家居用品": "耐用消费品与服装", "家电零部件Ⅱ": "耐用消费品与服装", "小家电": "耐用消费品与服装",
    "厨卫电器": "耐用消费品与服装", "白色家电": "耐用消费品与服装", "黑色家电": "耐用消费品与服装",
    "其他家电Ⅱ": "耐用消费品与服装", "照明设备Ⅱ": "耐用消费品与服装",
    "服装家纺": "耐用消费品与服装", "纺织制造": "耐用消费品与服装",
    "文娱用品": "耐用消费品与服装", "饰品": "耐用消费品与服装",
    "教育": "消费者服务", "旅游及景区": "消费者服务", "酒店餐饮": "消费者服务", "体育Ⅱ": "消费者服务",
    "一般零售": "可选消费零售", "专业连锁Ⅱ": "可选消费零售", "互联网电商": "可选消费零售",
    "旅游零售Ⅱ": "可选消费零售",
    "个护用品": "家庭与个人用品", "化妆品": "家庭与个人用品",
    "休闲食品": "食品饮料与烟草", "食品加工": "食品饮料与烟草", "饮料乳品": "食品饮料与烟草",
    "调味发酵品Ⅱ": "食品饮料与烟草", "白酒Ⅱ": "食品饮料与烟草", "非白酒": "食品饮料与烟草",
    "农产品加工": "食品饮料与烟草", "养殖业": "食品饮料与烟草", "种植业": "食品饮料与烟草",
    "渔业": "食品饮料与烟草", "农业综合Ⅱ": "食品饮料与烟草", "饲料": "食品饮料与烟草",
    "专业工程": "资本品", "专用设备": "资本品", "通用设备": "资本品", "工程机械": "资本品",
    "自动化设备": "资本品", "轨交设备Ⅱ": "资本品", "航天装备Ⅱ": "资本品", "航空装备Ⅱ": "资本品",
    "航海装备Ⅱ": "资本品", "地面兵装Ⅱ": "资本品", "军工电子Ⅱ": "资本品",
    "基础建设": "资本品", "房屋建设Ⅱ": "资本品", "装修装饰Ⅱ": "资本品",
    "电机Ⅱ": "资本品", "电池": "资本品", "光伏设备": "资本品", "风电设备": "资本品",
    "电网设备": "资本品", "其他电源设备Ⅱ": "资本品", "环保设备Ⅱ": "资本品",
    "综合Ⅱ": "资本品", "贸易Ⅱ": "资本品",
    "专业服务": "商业和专业服务", "工程咨询服务Ⅱ": "商业和专业服务", "环境治理": "商业和专业服务",
    "物流": "运输", "航空机场": "运输", "航运港口": "运输", "铁路公路": "运输",
    "煤炭开采": "能源", "焦炭Ⅱ": "能源", "油气开采Ⅱ": "能源", "油服工程": "能源", "炼化及贸易": "能源",
    "化学制品": "原材料", "化学原料": "原材料", "化学纤维": "原材料", "农化制品": "原材料",
    "塑料": "原材料", "橡胶": "原材料", "普钢": "原材料", "特钢Ⅱ": "原材料", "冶钢原料": "原材料",
    "工业金属": "原材料", "小金属": "原材料", "能源金属": "原材料", "贵金属": "原材料",
    "金属新材料": "原材料", "非金属材料Ⅱ": "原材料", "水泥": "原材料", "玻璃玻纤": "原材料",
    "装修建材": "原材料", "造纸": "原材料", "林业Ⅱ": "原材料", "包装印刷": "原材料",
    "电子化学品Ⅱ": "原材料",
    "电力": "公用事业", "燃气Ⅱ": "公用事业",
    "房地产开发": "房地产", "房地产服务": "房地产",
}

# 港股(东财行情列表行业)-> GICS 二级
HK2GRP = {
    "软件服务": "软件与服务", "资讯科技器材": "技术硬件与设备", "半导体": "半导体与半导体设备",
    "电讯": "电信服务", "媒体及娱乐": "媒体与娱乐",
    "药品及生物科技": "制药与生物科技", "其他医疗保健": "医疗保健设备与服务",
    "银行": "银行", "保险": "保险", "其他金融": "金融服务",
    "地产": "房地产", "公用事业": "公用事业",
    "石油及天然气": "能源", "煤炭": "能源",
    "一般金属及矿石": "原材料", "原材料": "原材料", "黄金及贵金属": "原材料",
    "工业工程": "资本品", "建筑": "资本品", "综合企业": "资本品",
    "工用支援": "商业和专业服务", "支援服务": "商业和专业服务",
    "工用运输": "运输",
    "汽车": "汽车与汽车零部件",
    "家庭电器及用品": "耐用消费品与服装", "纺织及服饰": "耐用消费品与服装",
    "旅游及消闲设施": "消费者服务",
    "专业零售": "可选消费零售",
    "消费者主要零售商": "日常消费零售",
    "食物饮品": "食品饮料与烟草", "农业产品": "食品饮料与烟草",
}

# 美股(纳斯达克 screener 行业,英文)-> GICS 二级
NAS2GRP = {
    "Accident &Health Insurance": "保险", "Advertising": "媒体与娱乐", "Aerospace": "资本品",
    "Agricultural Chemicals": "原材料", "Air Freight/Delivery Services": "运输", "Aluminum": "原材料",
    "Apparel": "耐用消费品与服装", "Auto & Home Supply Stores": "可选消费零售",
    "Auto Manufacturing": "汽车与汽车零部件", "Auto Parts:O.E.M.": "汽车与汽车零部件",
    "Automotive Aftermarket": "汽车与汽车零部件", "Banks": "银行",
    "Beverages (Production/Distribution)": "食品饮料与烟草",
    "Biotechnology: Biological Products (No Diagnostic Substances)": "制药与生物科技",
    "Biotechnology: Commercial Physical & Biological Resarch": "制药与生物科技",
    "Biotechnology: Electromedical & Electrotherapeutic Apparatus": "医疗保健设备与服务",
    "Biotechnology: In Vitro & In Vivo Diagnostic Substances": "医疗保健设备与服务",
    "Biotechnology: Laboratory Analytical Instruments": "医疗保健设备与服务",
    "Biotechnology: Pharmaceutical Preparations": "制药与生物科技",
    "Blank Checks": "金融服务", "Books": "媒体与娱乐", "Broadcasting": "媒体与娱乐",
    "Building Materials": "原材料", "Building Products": "资本品", "Building operators": "房地产",
    "Business Services": "商业和专业服务", "Cable & Other Pay Television Services": "媒体与娱乐",
    "Catalog/Specialty Distribution": "可选消费零售", "Clothing/Shoe/Accessory Stores": "可选消费零售",
    "Coal Mining": "能源", "Commercial Banks": "银行",
    "Computer Communications Equipment": "技术硬件与设备", "Computer Manufacturing": "技术硬件与设备",
    "Computer Software: Prepackaged Software": "软件与服务",
    "Computer Software: Programming Data Processing": "软件与服务",
    "Computer peripheral equipment": "技术硬件与设备",
    "Construction/Ag Equipment/Trucks": "资本品",
    "Consumer Electronics/Appliances": "耐用消费品与服装",
    "Consumer Electronics/Video Chains": "可选消费零售", "Consumer Specialties": "耐用消费品与服装",
    "Containers/Packaging": "原材料", "Department/Specialty Retail Stores": "可选消费零售",
    "Diversified Commercial Services": "商业和专业服务", "Diversified Electronic Products": "技术硬件与设备",
    "Diversified Financial Services": "金融服务", "Durable Goods": "耐用消费品与服装",
    "EDP Services": "软件与服务", "Electric Utilities: Central": "公用事业",
    "Electrical Products": "资本品", "Electronic Components": "技术硬件与设备",
    "Electronics Distribution": "技术硬件与设备", "Engineering & Construction": "资本品",
    "Environmental Services": "商业和专业服务", "Farming/Seeds/Milling": "食品饮料与烟草",
    "Finance Companies": "金融服务", "Finance/Investors Services": "金融服务",
    "Finance: Consumer Services": "金融服务", "Fluid Controls": "资本品",
    "Food Chains": "日常消费零售", "Food Distributors": "日常消费零售", "Forest Products": "原材料",
    "Garments and Clothing": "耐用消费品与服装",
    "General Bldg Contractors - Nonresidential Bldgs": "资本品",
    "Home Furnishings": "耐用消费品与服装", "Homebuilding": "耐用消费品与服装",
    "Hospital/Nursing Management": "医疗保健设备与服务", "Hotels/Resorts": "消费者服务",
    "Industrial Machinery/Components": "资本品", "Industrial Specialties": "原材料",
    "Integrated Freight & Logistics": "运输", "Integrated oil Companies": "能源",
    "Investment Bankers/Brokers/Service": "金融服务", "Investment Managers": "金融服务",
    "Life Insurance": "保险", "Major Banks": "银行", "Major Chemicals": "原材料",
    "Managed Health Care": "医疗保健设备与服务", "Marine Transportation": "运输",
    "Meat/Poultry/Fish": "食品饮料与烟草", "Medical Electronics": "医疗保健设备与服务",
    "Medical Specialities": "医疗保健设备与服务", "Medical/Dental Instruments": "医疗保健设备与服务",
    "Medical/Nursing Services": "医疗保健设备与服务",
    "Medicinal Chemicals and Botanical Products": "制药与生物科技",
    "Metal Fabrications": "资本品", "Metal Mining": "原材料", "Military/Government/Technical": "资本品",
    "Mining & Quarrying of Nonmetallic Minerals (No Fuels)": "原材料",
    "Misc Corporate Leasing Services": "商业和专业服务",
    "Misc Health and Biotechnology Services": "医疗保健设备与服务",
    "Miscellaneous manufacturing industries": "资本品", "Motor Vehicles": "汽车与汽车零部件",
    "Movies/Entertainment": "媒体与娱乐", "Multi-Sector Companies": "资本品",
    "Natural Gas Distribution": "公用事业", "Newspapers/Magazines": "媒体与娱乐",
    "Office Equipment/Supplies/Services": "商业和专业服务",
    "Oil & Gas Production": "能源", "Oil Refining/Marketing": "能源",
    "Oil and Gas Field Machinery": "能源", "Oil/Gas Transmission": "能源",
    "Oilfield Services/Equipment": "能源", "Ophthalmic Goods": "医疗保健设备与服务",
    "Ordnance And Accessories": "资本品", "Other Consumer Services": "消费者服务",
    "Other Metals and Minerals": "原材料", "Other Pharmaceuticals": "制药与生物科技",
    "Other Specialty Stores": "可选消费零售", "Other Transportation": "运输",
    "Package Goods/Cosmetics": "家庭与个人用品", "Packaged Foods": "食品饮料与烟草",
    "Paints/Coatings": "原材料", "Paper": "原材料",
    "Pharmaceuticals and Biotechnology": "制药与生物科技", "Plastic Products": "原材料",
    "Pollution Control Equipment": "资本品", "Power Generation": "公用事业",
    "Precious Metals": "原材料", "Precision Instruments": "技术硬件与设备",
    "Professional Services": "商业和专业服务", "Professional and commerical equipment": "技术硬件与设备",
    "Property-Casualty Insurers": "保险", "Publishing": "媒体与娱乐",
    "RETAIL: Building Materials": "可选消费零售",
    "Radio And Television Broadcasting And Communications Equipment": "技术硬件与设备",
    "Railroads": "运输", "Real Estate": "房地产", "Real Estate Investment Trusts": "房地产",
    "Recreational Games/Products/Toys": "耐用消费品与服装", "Rental/Leasing Companies": "商业和专业服务",
    "Restaurants": "消费者服务", "Retail-Auto Dealers and Gas Stations": "可选消费零售",
    "Retail-Drug Stores and Proprietary Stores": "日常消费零售",
    "Retail: Computer Software & Peripheral Equipment": "可选消费零售",
    "Savings Institutions": "银行", "Semiconductors": "半导体与半导体设备",
    "Services-Misc. Amusement & Recreation": "消费者服务", "Shoe Manufacturing": "耐用消费品与服装",
    "Specialty Chemicals": "原材料", "Specialty Foods": "食品饮料与烟草", "Specialty Insurers": "保险",
    "Steel/Iron Ore": "原材料", "Telecommunications Equipment": "技术硬件与设备",
    "Textiles": "耐用消费品与服装", "Tobacco": "食品饮料与烟草", "Tools/Hardware": "资本品",
    "Transportation Services": "运输", "Trucking Freight/Courier Services": "运输",
    "Trusts Except Educational Religious and Charitable": "金融服务",
    "Water Sewer Pipeline Comm & Power Line Construction": "资本品", "Water Supply": "公用事业",
    "Wholesale Distributors": "资本品",
}


# 重点公司逐家覆盖(按 GICS 官方归类,修正来源分类的明显偏差)
US_OVERRIDE = {
    "V": ("金融", "金融服务"), "MA": ("金融", "金融服务"), "PYPL": ("金融", "金融服务"),
    "NFLX": ("通讯服务", "媒体与娱乐"), "SPOT": ("通讯服务", "媒体与娱乐"),
    "META": ("通讯服务", "媒体与娱乐"), "DIS": ("通讯服务", "媒体与娱乐"),
}
HK_OVERRIDE = {
    "00700": ("通讯服务", "媒体与娱乐"),   # 腾讯
    "09999": ("通讯服务", "媒体与娱乐"),   # 网易
    "01024": ("通讯服务", "媒体与娱乐"),   # 快手
    "09888": ("通讯服务", "媒体与娱乐"),   # 百度
    "09626": ("通讯服务", "媒体与娱乐"),   # 哔哩哔哩
}
A_OVERRIDE = {}


def sec_g_for_a(ind, code=None):
    """A股:原生行业 -> (一级, 二级)。"""
    if code and code in A_OVERRIDE:
        return A_OVERRIDE[code]
    g = A2GRP.get(ind or "", "")
    return (GRP2SEC.get(g, ""), g)


def sec_g_for_hk(ind, code=None):
    if code and code in HK_OVERRIDE:
        return HK_OVERRIDE[code]
    g = HK2GRP.get(ind or "", "")
    return (GRP2SEC.get(g, ""), g)


def sec_g_for_us(em_sector, nasdaq_industry, code=None):
    """美股:东财一级大类 + 纳斯达克细分行业 -> (一级, 二级)。
    细分行业映射出的二级若与东财大类冲突,以东财大类为准取其缺省二级
    (纳斯达克自家分类有不少与 GICS 不一致,东财大类是 GICS 口径,更可靠)。"""
    if code and code in US_OVERRIDE:
        return US_OVERRIDE[code]
    sec = US_SECTOR_ALIAS.get(em_sector or "", em_sector or "")
    g = NAS2GRP.get((nasdaq_industry or "").strip(), "")
    if not sec:
        return (GRP2SEC.get(g, ""), g)
    if not g or GRP2SEC.get(g) != sec:
        if sec == "通讯服务":
            # 通讯服务内部歧义:互联网/软件/媒体类归媒体与娱乐,其余(运营商)归电信服务
            g = "媒体与娱乐" if g in ("软件与服务", "可选消费零售", "消费者服务") else "电信服务"
        else:
            g = SECTOR_DEFAULT_GRP.get(sec, "")
    return (sec, g)
