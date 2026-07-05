"""82 个纯价量 alpha。逐字翻译自 Kakushadze (2015) 附录 A。"""
import pandas as pd
from alpha101 import operators as op
from alpha101 import data


def alpha_1(P):
    """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    inner = op.ts_stddev(P["returns"], 20).where(P["returns"] < 0, P["close"])
    x = op.signedpower(inner, 2.0)
    return op.rank(op.ts_argmax(x, 5)) - 0.5


def alpha_2(P):
    """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    return -1 * op.correlation(
        op.rank(op.delta(op.log_(P["volume"]), 2)),
        op.rank((P["close"] - P["open"]) / P["open"]),
        6,
    )


def alpha_3(P):
    """(-1 * correlation(rank(open), rank(volume), 10))"""
    return -1 * op.correlation(op.rank(P["open"]), op.rank(P["volume"]), 10)


def alpha_4(P):
    """(-1 * Ts_Rank(rank(low), 9))"""
    return -1 * op.ts_rank(op.rank(P["low"]), 9)


def alpha_5(P):
    """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    return op.rank(P["open"] - (op.ts_sum(P["vwap"], 10) / 10)) * (
        -1 * op.abs_(op.rank(P["close"] - P["vwap"]))
    )


def alpha_6(P):
    """(-1 * correlation(open, volume, 10))"""
    return -1 * op.correlation(P["open"], P["volume"], 10)


def alpha_7(P):
    """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
    adv20 = data.adv(P, 20)
    delta7 = op.delta(P["close"], 7)
    then = (-1 * op.ts_rank(op.abs_(delta7), 60)) * op.sign_(delta7)
    cond = adv20 < P["volume"]
    return then.where(cond, -1.0)


def alpha_8(P):
    """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    x = op.ts_sum(P["open"], 5) * op.ts_sum(P["returns"], 5)
    return -1 * op.rank(x - op.delay(x, 10))


def alpha_9(P):
    """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
    d1 = op.delta(P["close"], 1)
    cond1 = 0 < op.ts_min(d1, 5)
    cond2 = op.ts_max(d1, 5) < 0
    inner = d1.where(cond2, -1 * d1)
    return d1.where(cond1, inner)


def alpha_10(P):
    """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    d1 = op.delta(P["close"], 1)
    cond1 = 0 < op.ts_min(d1, 4)
    cond2 = op.ts_max(d1, 4) < 0
    inner = d1.where(cond2, -1 * d1)
    return op.rank(d1.where(cond1, inner))


def alpha_11(P):
    """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    vc = P["vwap"] - P["close"]
    return (op.rank(op.ts_max(vc, 3)) + op.rank(op.ts_min(vc, 3))) * op.rank(
        op.delta(P["volume"], 3)
    )


def alpha_12(P):
    """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    return op.sign_(op.delta(P["volume"], 1)) * (-1 * op.delta(P["close"], 1))


def alpha_13(P):
    """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    return -1 * op.rank(op.covariance(op.rank(P["close"]), op.rank(P["volume"]), 5))


def alpha_14(P):
    """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    return (-1 * op.rank(op.delta(P["returns"], 3))) * op.correlation(
        P["open"], P["volume"], 10
    )


def alpha_15(P):
    """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    return -1 * op.ts_sum(
        op.rank(op.correlation(op.rank(P["high"]), op.rank(P["volume"]), 3)), 3
    )


def alpha_16(P):
    """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    return -1 * op.rank(op.covariance(op.rank(P["high"]), op.rank(P["volume"]), 5))


def alpha_17(P):
    """(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    adv20 = data.adv(P, 20)
    return (
        (-1 * op.rank(op.ts_rank(P["close"], 10)))
        * op.rank(op.delta(op.delta(P["close"], 1), 1))
    ) * op.rank(op.ts_rank(P["volume"] / adv20, 5))


def alpha_18(P):
    """(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
    co = P["close"] - P["open"]
    return -1 * op.rank(
        (op.ts_stddev(op.abs_(co), 5) + co) + op.correlation(P["close"], P["open"], 10)
    )


def alpha_19(P):
    """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    return (
        -1 * op.sign_((P["close"] - op.delay(P["close"], 7)) + op.delta(P["close"], 7))
    ) * (1 + op.rank(1 + op.ts_sum(P["returns"], 250)))


def alpha_20(P):
    """(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    return (
        (-1 * op.rank(P["open"] - op.delay(P["high"], 1)))
        * op.rank(P["open"] - op.delay(P["close"], 1))
    ) * op.rank(P["open"] - op.delay(P["low"], 1))


def alpha_21(P):
    """
    (((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1)
    : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1
    : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1)))
    """
    adv20 = data.adv(P, 20)
    m8 = op.ts_sum(P["close"], 8) / 8
    s8 = op.ts_stddev(P["close"], 8)
    m2 = op.ts_sum(P["close"], 2) / 2
    cond1 = (m8 + s8) < m2
    cond2 = m2 < (m8 - s8)
    vol_ratio = P["volume"] / adv20
    cond3 = (1 < vol_ratio) | (vol_ratio == 1)
    innermost = pd.DataFrame(1.0, index=P["close"].index, columns=P["close"].columns).where(
        cond3, -1.0
    )
    mid = pd.DataFrame(1.0, index=P["close"].index, columns=P["close"].columns).where(
        cond2, innermost
    )
    return pd.DataFrame(-1.0, index=P["close"].index, columns=P["close"].columns).where(
        cond1, mid
    )


def alpha_22(P):
    """(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
    return -1 * (
        op.delta(op.correlation(P["high"], P["volume"], 5), 5)
        * op.rank(op.ts_stddev(P["close"], 20))
    )


def alpha_23(P):
    """(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
    cond = (op.ts_sum(P["high"], 20) / 20) < P["high"]
    then = -1 * op.delta(P["high"], 2)
    zero = pd.DataFrame(0.0, index=P["high"].index, columns=P["high"].columns)
    return then.where(cond, zero)


def alpha_24(P):
    """
    ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05))
    ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
    """
    ma100 = op.ts_sum(P["close"], 100) / 100
    ratio = op.delta(ma100, 100) / op.delay(P["close"], 100)
    cond = (ratio < 0.05) | (ratio == 0.05)
    then = -1 * (P["close"] - op.ts_min(P["close"], 100))
    other = -1 * op.delta(P["close"], 3)
    return then.where(cond, other)


def alpha_25(P):
    """rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
    adv20 = data.adv(P, 20)
    return op.rank((((-1 * P["returns"]) * adv20) * P["vwap"]) * (P["high"] - P["close"]))


def alpha_26(P):
    """(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    return -1 * op.ts_max(
        op.correlation(op.ts_rank(P["volume"], 5), op.ts_rank(P["high"], 5), 5), 3
    )


def alpha_27(P):
    """(0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1"""
    x = op.rank(
        op.ts_sum(op.correlation(op.rank(P["volume"]), op.rank(P["vwap"]), 6), 2) / 2.0
    )
    cond = 0.5 < x
    ones = pd.DataFrame(1.0, index=x.index, columns=x.columns)
    return (-1 * ones).where(cond, ones)


def alpha_28(P):
    """scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    adv20 = data.adv(P, 20)
    return op.scale(
        (op.correlation(adv20, P["low"], 5) + ((P["high"] + P["low"]) / 2)) - P["close"]
    )


def alpha_29(P):
    """
    (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5)
    + ts_rank(delay((-1 * returns), 6), 5))
    """
    inner = op.rank(op.rank(-1 * op.rank(op.delta(P["close"] - 1, 5))))
    inner = op.ts_min(inner, 2)
    inner = op.ts_sum(inner, 1)
    inner = op.log_(inner)
    inner = op.scale(inner)
    inner = op.rank(op.rank(inner))
    inner = op.ts_product(inner, 1)
    return op.ts_min(inner, 5) + op.ts_rank(op.delay(-1 * P["returns"], 6), 5)


def alpha_30(P):
    """
    (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3))))))
    * sum(volume, 5)) / sum(volume, 20))
    """
    close = P["close"]
    s = (
        op.sign_(close - op.delay(close, 1))
        + op.sign_(op.delay(close, 1) - op.delay(close, 2))
        + op.sign_(op.delay(close, 2) - op.delay(close, 3))
    )
    return (
        (1.0 - op.rank(s)) * op.ts_sum(P["volume"], 5)
    ) / op.ts_sum(P["volume"], 20)


def alpha_31(P):
    """
    ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3))))
    + sign(scale(correlation(adv20, low, 12))))
    """
    adv20 = data.adv(P, 20)
    term1 = op.rank(
        op.rank(op.rank(op.decay_linear(-1 * op.rank(op.rank(op.delta(P["close"], 10))), 10)))
    )
    term2 = op.rank(-1 * op.delta(P["close"], 3))
    term3 = op.sign_(op.scale(op.correlation(adv20, P["low"], 12)))
    return term1 + term2 + term3


def alpha_32(P):
    """(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
    return op.scale((op.ts_sum(P["close"], 7) / 7) - P["close"]) + (
        20 * op.scale(op.correlation(P["vwap"], op.delay(P["close"], 5), 230))
    )


def alpha_33(P):
    """rank((-1 * ((1 - (open / close))^1)))"""
    return op.rank(-1 * ((1 - (P["open"] / P["close"])) ** 1))


def alpha_34(P):
    """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    return op.rank(
        (1 - op.rank(op.ts_stddev(P["returns"], 2) / op.ts_stddev(P["returns"], 5)))
        + (1 - op.rank(op.delta(P["close"], 1)))
    )


def alpha_35(P):
    """((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    return (
        op.ts_rank(P["volume"], 32)
        * (1 - op.ts_rank((P["close"] + P["high"]) - P["low"], 16))
    ) * (1 - op.ts_rank(P["returns"], 32))


def alpha_36(P):
    """
    (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close))))
    + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6))))
    + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    """
    adv20 = data.adv(P, 20)
    t1 = 2.21 * op.rank(
        op.correlation(P["close"] - P["open"], op.delay(P["volume"], 1), 15)
    )
    t2 = 0.7 * op.rank(P["open"] - P["close"])
    t3 = 0.73 * op.rank(op.ts_rank(op.delay(-1 * P["returns"], 6), 5))
    t4 = op.rank(op.abs_(op.correlation(P["vwap"], adv20, 6)))
    t5 = 0.6 * op.rank(
        ((op.ts_sum(P["close"], 200) / 200) - P["open"]) * (P["close"] - P["open"])
    )
    return t1 + t2 + t3 + t4 + t5


def alpha_37(P):
    """(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    return op.rank(
        op.correlation(op.delay(P["open"] - P["close"], 1), P["close"], 200)
    ) + op.rank(P["open"] - P["close"])


def alpha_38(P):
    """((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
    return (-1 * op.rank(op.ts_rank(P["close"], 10))) * op.rank(P["close"] / P["open"])


def alpha_39(P):
    """((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    adv20 = data.adv(P, 20)
    return (
        -1
        * op.rank(
            op.delta(P["close"], 7)
            * (1 - op.rank(op.decay_linear(P["volume"] / adv20, 9)))
        )
    ) * (1 + op.rank(op.ts_sum(P["returns"], 250)))


def alpha_40(P):
    """((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
    return (-1 * op.rank(op.ts_stddev(P["high"], 10))) * op.correlation(
        P["high"], P["volume"], 10
    )


def alpha_41(P):
    """(((high * low)^0.5) - vwap)"""
    return (P["high"] * P["low"]) ** 0.5 - P["vwap"]


def alpha_42(P):
    """(rank((vwap - close)) / rank((vwap + close)))"""
    return op.rank(P["vwap"] - P["close"]) / op.rank(P["vwap"] + P["close"])


def alpha_43(P):
    """(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
    adv20 = data.adv(P, 20)
    return op.ts_rank(P["volume"] / adv20, 20) * op.ts_rank(-1 * op.delta(P["close"], 7), 8)


def alpha_44(P):
    """(-1 * correlation(high, rank(volume), 5))"""
    return -1 * op.correlation(P["high"], op.rank(P["volume"]), 5)


def alpha_45(P):
    """
    (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2))
    * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    """
    return -1 * (
        (op.rank(op.ts_sum(op.delay(P["close"], 5), 20) / 20)
         * op.correlation(P["close"], P["volume"], 2))
        * op.rank(op.correlation(op.ts_sum(P["close"], 5), op.ts_sum(P["close"], 20), 2))
    )


def alpha_46(P):
    """
    ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)))
    ? (-1 * 1)
    : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0)
    ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
    """
    close = P["close"]
    x = ((op.delay(close, 20) - op.delay(close, 10)) / 10) - (
        (op.delay(close, 10) - close) / 10
    )
    cond1 = 0.25 < x
    cond2 = x < 0
    other = (-1 * 1) * (close - op.delay(close, 1))
    inner = pd.DataFrame(1.0, index=close.index, columns=close.columns).where(cond2, other)
    return pd.DataFrame(-1.0, index=close.index, columns=close.columns).where(cond1, inner)


def alpha_47(P):
    """
    ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5)))
    - rank((vwap - delay(vwap, 5))))
    """
    adv20 = data.adv(P, 20)
    return (
        ((op.rank(1 / P["close"]) * P["volume"]) / adv20)
        * ((P["high"] * op.rank(P["high"] - P["close"])) / (op.ts_sum(P["high"], 5) / 5))
    ) - op.rank(P["vwap"] - op.delay(P["vwap"], 5))


def alpha_49(P):
    """
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1))
    ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    close = P["close"]
    x = ((op.delay(close, 20) - op.delay(close, 10)) / 10) - (
        (op.delay(close, 10) - close) / 10
    )
    cond = x < (-1 * 0.1)
    other = (-1 * 1) * (close - op.delay(close, 1))
    ones = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    return ones.where(cond, other)


def alpha_50(P):
    """(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    return -1 * op.ts_max(
        op.rank(op.correlation(op.rank(P["volume"]), op.rank(P["vwap"]), 5)), 5
    )


def alpha_51(P):
    """
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05))
    ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    close = P["close"]
    x = ((op.delay(close, 20) - op.delay(close, 10)) / 10) - (
        (op.delay(close, 10) - close) / 10
    )
    cond = x < (-1 * 0.05)
    other = (-1 * 1) * (close - op.delay(close, 1))
    ones = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    return ones.where(cond, other)


def alpha_52(P):
    """
    ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220)))
    * ts_rank(volume, 5))
    """
    tsmin_low5 = op.ts_min(P["low"], 5)
    return (
        ((-1 * tsmin_low5) + op.delay(tsmin_low5, 5))
        * op.rank((op.ts_sum(P["returns"], 240) - op.ts_sum(P["returns"], 20)) / 220)
    ) * op.ts_rank(P["volume"], 5)


def alpha_53(P):
    """(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
    x = ((P["close"] - P["low"]) - (P["high"] - P["close"])) / (P["close"] - P["low"])
    return -1 * op.delta(x, 9)


def alpha_54(P):
    """((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
    num = -1 * ((P["low"] - P["close"]) * (P["open"] ** 5))
    den = (P["low"] - P["high"]) * (P["close"] ** 5)
    return num / den


def alpha_55(P):
    """(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
    x = (P["close"] - op.ts_min(P["low"], 12)) / (
        op.ts_max(P["high"], 12) - op.ts_min(P["low"], 12)
    )
    return -1 * op.correlation(op.rank(x), op.rank(P["volume"]), 6)


def alpha_57(P):
    """(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
    return 0 - (
        1
        * ((P["close"] - P["vwap"]) / op.decay_linear(op.rank(op.ts_argmax(P["close"], 30)), 2))
    )


def alpha_60(P):
    """
    (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume))))
    - scale(rank(ts_argmax(close, 10))))))
    """
    x = (((P["close"] - P["low"]) - (P["high"] - P["close"])) / (P["high"] - P["low"])) * P[
        "volume"
    ]
    return 0 - (
        1
        * (
            (2 * op.scale(op.rank(x)))
            - op.scale(op.rank(op.ts_argmax(P["close"], 10)))
        )
    )


def alpha_61(P):
    """(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"""
    adv180 = data.adv(P, 180)
    lhs = op.rank(P["vwap"] - op.ts_min(P["vwap"], 16.1219))
    rhs = op.rank(op.correlation(P["vwap"], adv180, 17.9282))
    return (lhs < rhs).astype(float)


def alpha_62(P):
    """
    ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009))
    < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    """
    adv20 = data.adv(P, 20)
    lhs = op.rank(op.correlation(P["vwap"], op.ts_sum(adv20, 22.4101), 9.91009))
    inner_cond = ((op.rank(P["open"]) + op.rank(P["open"]))
                  < (op.rank((P["high"] + P["low"]) / 2) + op.rank(P["high"])))
    rhs = op.rank(inner_cond.astype(float))
    return (lhs < rhs).astype(float) * -1


def alpha_64(P):
    """
    ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208))
    < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    """
    adv120 = data.adv(P, 120)
    lhs = op.rank(
        op.correlation(
            op.ts_sum(P["open"] * 0.178404 + P["low"] * (1 - 0.178404), 12.7054),
            op.ts_sum(adv120, 12.7054),
            16.6208,
        )
    )
    rhs = op.rank(
        op.delta(
            ((P["high"] + P["low"]) / 2) * 0.178404 + P["vwap"] * (1 - 0.178404), 3.69741
        )
    )
    return (lhs < rhs).astype(float) * -1


def alpha_65(P):
    """
    ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374))
    < rank((open - ts_min(open, 13.635)))) * -1)
    """
    adv60 = data.adv(P, 60)
    lhs = op.rank(
        op.correlation(
            P["open"] * 0.00817205 + P["vwap"] * (1 - 0.00817205),
            op.ts_sum(adv60, 8.6911),
            6.40374,
        )
    )
    rhs = op.rank(P["open"] - op.ts_min(P["open"], 13.635))
    return (lhs < rhs).astype(float) * -1


def alpha_66(P):
    """
    ((rank(decay_linear(delta(vwap, 3.51013), 7.23052))
    + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    """
    t1 = op.rank(op.decay_linear(op.delta(P["vwap"], 3.51013), 7.23052))
    lowblend = P["low"] * 0.96633 + P["low"] * (1 - 0.96633)
    t2 = op.ts_rank(
        op.decay_linear(
            (lowblend - P["vwap"]) / (P["open"] - ((P["high"] + P["low"]) / 2)), 11.4157
        ),
        6.72611,
    )
    return (t1 + t2) * -1


def alpha_68(P):
    """
    ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333)
    < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    """
    adv15 = data.adv(P, 15)
    lhs = op.ts_rank(op.correlation(op.rank(P["high"]), op.rank(adv15), 8.91644), 13.9333)
    rhs = op.rank(op.delta(P["close"] * 0.518371 + P["low"] * (1 - 0.518371), 1.06157))
    return (lhs < rhs).astype(float) * -1


def alpha_71(P):
    """
    max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948),
    Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
    """
    adv180 = data.adv(P, 180)
    a = op.ts_rank(
        op.decay_linear(
            op.correlation(op.ts_rank(P["close"], 3.43976), op.ts_rank(adv180, 12.0647), 18.0175),
            4.20501,
        ),
        15.6948,
    )
    b = op.ts_rank(
        op.decay_linear(
            op.rank((P["low"] + P["open"]) - (P["vwap"] + P["vwap"])) ** 2, 16.4662
        ),
        4.4388,
    )
    return op.ew_max(a, b)


def alpha_72(P):
    """
    (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519))
    / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
    """
    adv40 = data.adv(P, 40)
    num = op.rank(
        op.decay_linear(op.correlation((P["high"] + P["low"]) / 2, adv40, 8.93345), 10.1519)
    )
    den = op.rank(
        op.decay_linear(
            op.correlation(op.ts_rank(P["vwap"], 3.72469), op.ts_rank(P["volume"], 18.5188), 6.86671),
            2.95011,
        )
    )
    return num / den


def alpha_73(P):
    """
    (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608)
    / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    """
    a = op.rank(op.decay_linear(op.delta(P["vwap"], 4.72775), 2.91864))
    blend = P["open"] * 0.147155 + P["low"] * (1 - 0.147155)
    b = op.ts_rank(
        op.decay_linear((op.delta(blend, 2.03608) / blend) * -1, 3.33829), 16.7411
    )
    return op.ew_max(a, b) * -1


def alpha_74(P):
    """
    ((rank(correlation(close, sum(adv30, 37.4843), 15.1365))
    < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
    """
    adv30 = data.adv(P, 30)
    lhs = op.rank(op.correlation(P["close"], op.ts_sum(adv30, 37.4843), 15.1365))
    rhs = op.rank(
        op.correlation(
            op.rank(P["high"] * 0.0261661 + P["vwap"] * (1 - 0.0261661)),
            op.rank(P["volume"]),
            11.4791,
        )
    )
    return (lhs < rhs).astype(float) * -1


def alpha_75(P):
    """(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))"""
    adv50 = data.adv(P, 50)
    lhs = op.rank(op.correlation(P["vwap"], P["volume"], 4.24304))
    rhs = op.rank(op.correlation(op.rank(P["low"]), op.rank(adv50), 12.4413))
    return (lhs < rhs).astype(float)


def alpha_77(P):
    """
    min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
    rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    """
    adv40 = data.adv(P, 40)
    a = op.rank(
        op.decay_linear(
            (((P["high"] + P["low"]) / 2) + P["high"]) - (P["vwap"] + P["high"]), 20.0451
        )
    )
    b = op.rank(
        op.decay_linear(op.correlation((P["high"] + P["low"]) / 2, adv40, 3.1614), 5.64125)
    )
    return op.ew_min(a, b)


def alpha_78(P):
    """
    (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))
    ^ rank(correlation(rank(vwap), rank(volume), 5.77492)))
    """
    adv40 = data.adv(P, 40)
    lhs = op.rank(
        op.correlation(
            op.ts_sum(P["low"] * 0.352233 + P["vwap"] * (1 - 0.352233), 19.7428),
            op.ts_sum(adv40, 19.7428),
            6.83313,
        )
    )
    rhs = op.rank(op.correlation(op.rank(P["vwap"]), op.rank(P["volume"]), 5.77492))
    return lhs ** rhs


def alpha_81(P):
    """
    ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655)))
    < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    """
    adv10 = data.adv(P, 10)
    inner = op.rank(op.correlation(P["vwap"], op.ts_sum(adv10, 49.6054), 8.47743)) ** 4
    lhs = op.rank(op.log_(op.ts_product(op.rank(inner), 14.9655)))
    rhs = op.rank(op.correlation(op.rank(P["vwap"]), op.rank(P["volume"]), 5.07914))
    return (lhs < rhs).astype(float) * -1


def alpha_83(P):
    """
    ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume)))
    / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    """
    hl_ratio = (P["high"] - P["low"]) / (op.ts_sum(P["close"], 5) / 5)
    num = op.rank(op.delay(hl_ratio, 2)) * op.rank(op.rank(P["volume"]))
    den = hl_ratio / (P["vwap"] - P["close"])
    return num / den


def alpha_84(P):
    """SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))"""
    return op.signedpower(
        op.ts_rank(P["vwap"] - op.ts_max(P["vwap"], 15.3217), 20.7127),
        op.delta(P["close"], 4.96796),
    )


def alpha_85(P):
    """
    (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))
    ^ rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
    """
    adv30 = data.adv(P, 30)
    lhs = op.rank(
        op.correlation(P["high"] * 0.876703 + P["close"] * (1 - 0.876703), adv30, 9.61331)
    )
    rhs = op.rank(
        op.correlation(
            op.ts_rank((P["high"] + P["low"]) / 2, 3.70596),
            op.ts_rank(P["volume"], 10.1595),
            7.11408,
        )
    )
    return lhs ** rhs


def alpha_86(P):
    """
    ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195)
    < rank(((open + close) - (vwap + open)))) * -1)
    """
    adv20 = data.adv(P, 20)
    lhs = op.ts_rank(op.correlation(P["close"], op.ts_sum(adv20, 14.7444), 6.00049), 20.4195)
    rhs = op.rank((P["open"] + P["close"]) - (P["vwap"] + P["open"]))
    return (lhs < rhs).astype(float) * -1


def alpha_88(P):
    """
    min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)),
    Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
    """
    adv60 = data.adv(P, 60)
    a = op.rank(
        op.decay_linear(
            (op.rank(P["open"]) + op.rank(P["low"])) - (op.rank(P["high"]) + op.rank(P["close"])),
            8.06882,
        )
    )
    b = op.ts_rank(
        op.decay_linear(
            op.correlation(op.ts_rank(P["close"], 8.44728), op.ts_rank(adv60, 20.6966), 8.01266),
            6.65053,
        ),
        2.61957,
    )
    return op.ew_min(a, b)


def alpha_92(P):
    """
    min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683),
    Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
    """
    adv30 = data.adv(P, 30)
    cond = (((P["high"] + P["low"]) / 2) + P["close"]) < (P["low"] + P["open"])
    a = op.ts_rank(op.decay_linear(cond.astype(float), 14.7221), 18.8683)
    b = op.ts_rank(
        op.decay_linear(op.correlation(op.rank(P["low"]), op.rank(adv30), 7.58555), 6.94024),
        6.80584,
    )
    return op.ew_min(a, b)


def alpha_94(P):
    """
    ((rank((vwap - ts_min(vwap, 11.5783)))
    ^ Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    """
    adv60 = data.adv(P, 60)
    lhs = op.rank(P["vwap"] - op.ts_min(P["vwap"], 11.5783))
    rhs = op.ts_rank(
        op.correlation(op.ts_rank(P["vwap"], 19.6462), op.ts_rank(adv60, 4.02992), 18.0926),
        2.70756,
    )
    return (lhs ** rhs) * -1


def alpha_95(P):
    """
    (rank((open - ts_min(open, 12.4105)))
    < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    """
    adv40 = data.adv(P, 40)
    lhs = op.rank(P["open"] - op.ts_min(P["open"], 12.4105))
    inner = (
        op.rank(
            op.correlation(
                op.ts_sum((P["high"] + P["low"]) / 2, 19.1351), op.ts_sum(adv40, 19.1351), 12.8742
            )
        )
        ** 5
    )
    rhs = op.ts_rank(inner, 11.7584)
    return (lhs < rhs).astype(float)


def alpha_96(P):
    """
    (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151),
    Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    """
    adv60 = data.adv(P, 60)
    a = op.ts_rank(
        op.decay_linear(op.correlation(op.rank(P["vwap"]), op.rank(P["volume"]), 3.83878), 4.16783),
        8.38151,
    )
    b = op.ts_rank(
        op.decay_linear(
            op.ts_argmax(
                op.correlation(op.ts_rank(P["close"], 7.45404), op.ts_rank(adv60, 4.13242), 3.65459),
                12.6556,
            ),
            14.0365,
        ),
        13.4143,
    )
    return op.ew_max(a, b) * -1


def alpha_98(P):
    """
    (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088))
    - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
    """
    adv5 = data.adv(P, 5)
    adv15 = data.adv(P, 15)
    t1 = op.rank(
        op.decay_linear(op.correlation(P["vwap"], op.ts_sum(adv5, 26.4719), 4.58418), 7.18088)
    )
    t2 = op.rank(
        op.decay_linear(
            op.ts_rank(
                op.ts_argmin(op.correlation(op.rank(P["open"]), op.rank(adv15), 20.8187), 8.62571),
                6.95668,
            ),
            8.07206,
        )
    )
    return t1 - t2


def alpha_99(P):
    """
    ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136))
    < rank(correlation(low, volume, 6.28259))) * -1)
    """
    adv60 = data.adv(P, 60)
    lhs = op.rank(
        op.correlation(op.ts_sum((P["high"] + P["low"]) / 2, 19.8975), op.ts_sum(adv60, 19.8975), 8.8136)
    )
    rhs = op.rank(op.correlation(P["low"], P["volume"], 6.28259))
    return (lhs < rhs).astype(float) * -1


def alpha_101(P):
    """((close - open) / ((high - low) + .001))"""
    return (P["close"] - P["open"]) / ((P["high"] - P["low"]) + 0.001)


# ============ 行业中性化 alpha(需 P["ind"]=Series(code->行业))============
# 论文 IndClass.sector/industry/subindustry 各级,这里统一映射到单一行业分类。

def alpha_48(P):
    """indneutralize((correlation(delta(close,1),delta(delay(close,1),1),250)*delta(close,1))/close, IndClass.subindustry) / sum((delta(close,1)/delay(close,1))^2, 250)"""
    num = op.indneutralize((op.correlation(op.delta(P["close"], 1), op.delta(op.delay(P["close"], 1), 1), 250) * op.delta(P["close"], 1)) / P["close"], P["ind"])
    den = op.ts_sum((op.delta(P["close"], 1) / op.delay(P["close"], 1)) ** 2, 250)
    return num / den


def alpha_58(P):
    """-1*Ts_Rank(decay_linear(correlation(IndNeutralize(vwap,IndClass.sector),volume,3.92795),7.89291),5.50322)"""
    return -1 * op.ts_rank(op.decay_linear(op.correlation(op.indneutralize(P["vwap"], P["ind"]), P["volume"], 3.92795), 7.89291), 5.50322)


def alpha_59(P):
    """-1*Ts_Rank(decay_linear(correlation(IndNeutralize((vwap*0.728317)+(vwap*(1-0.728317)),IndClass.industry),volume,4.25197),16.2289),8.19648)"""
    w = P["vwap"] * 0.728317 + P["vwap"] * (1 - 0.728317)
    return -1 * op.ts_rank(op.decay_linear(op.correlation(op.indneutralize(w, P["ind"]), P["volume"], 4.25197), 16.2289), 8.19648)


def alpha_63(P):
    """(rank(decay_linear(delta(IndNeutralize(close,IndClass.industry),2.25164),8.22237)) - rank(decay_linear(correlation((vwap*0.318108)+(open*(1-0.318108)),sum(adv180,37.2467),13.557),12.2883)))*-1"""
    a = op.rank(op.decay_linear(op.delta(op.indneutralize(P["close"], P["ind"]), 2.25164), 8.22237))
    b = op.rank(op.decay_linear(op.correlation(P["vwap"] * 0.318108 + P["open"] * (1 - 0.318108), op.ts_sum(data.adv(P, 180), 37.2467), 13.557), 12.2883))
    return (a - b) * -1


def alpha_67(P):
    """(rank(high-ts_min(high,2.14593))^rank(correlation(IndNeutralize(vwap,IndClass.sector),IndNeutralize(adv20,IndClass.subindustry),6.02936)))*-1"""
    a = op.rank(P["high"] - op.ts_min(P["high"], 2.14593))
    b = op.rank(op.correlation(op.indneutralize(P["vwap"], P["ind"]), op.indneutralize(data.adv(P, 20), P["ind"]), 6.02936))
    return (a ** b) * -1


def alpha_69(P):
    """(rank(ts_max(delta(IndNeutralize(vwap,IndClass.industry),2.72412),4.79344))^Ts_Rank(correlation((close*0.490655)+(vwap*(1-0.490655)),adv20,4.92416),9.0615))*-1"""
    a = op.rank(op.ts_max(op.delta(op.indneutralize(P["vwap"], P["ind"]), 2.72412), 4.79344))
    b = op.ts_rank(op.correlation(P["close"] * 0.490655 + P["vwap"] * (1 - 0.490655), data.adv(P, 20), 4.92416), 9.0615)
    return (a ** b) * -1


def alpha_70(P):
    """(rank(delta(vwap,1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry),adv50,17.8256),17.9171))*-1"""
    a = op.rank(op.delta(P["vwap"], 1.29456))
    b = op.ts_rank(op.correlation(op.indneutralize(P["close"], P["ind"]), data.adv(P, 50), 17.8256), 17.9171)
    return (a ** b) * -1


def alpha_76(P):
    """max(rank(decay_linear(delta(vwap,1.24383),11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low,IndClass.sector),adv81,8.14941),19.569),17.1543),19.383))*-1"""
    a = op.rank(op.decay_linear(op.delta(P["vwap"], 1.24383), 11.8259))
    b = op.ts_rank(op.decay_linear(op.ts_rank(op.correlation(op.indneutralize(P["low"], P["ind"]), data.adv(P, 81), 8.14941), 19.569), 17.1543), 19.383)
    return op.ew_max(a, b) * -1


def alpha_79(P):
    """rank(delta(IndNeutralize((close*0.60733)+(open*(1-0.60733)),IndClass.sector),1.23438)) < rank(correlation(Ts_Rank(vwap,3.60973),Ts_Rank(adv150,9.18637),14.6644))"""
    a = op.rank(op.delta(op.indneutralize(P["close"] * 0.60733 + P["open"] * (1 - 0.60733), P["ind"]), 1.23438))
    b = op.rank(op.correlation(op.ts_rank(P["vwap"], 3.60973), op.ts_rank(data.adv(P, 150), 9.18637), 14.6644))
    return (a < b).astype(float)


def alpha_80(P):
    """(rank(Sign(delta(IndNeutralize((open*0.868128)+(high*(1-0.868128)),IndClass.industry),4.04545)))^Ts_Rank(correlation(high,adv10,5.11456),5.53756))*-1"""
    a = op.rank(op.sign_(op.delta(op.indneutralize(P["open"] * 0.868128 + P["high"] * (1 - 0.868128), P["ind"]), 4.04545)))
    b = op.ts_rank(op.correlation(P["high"], data.adv(P, 10), 5.11456), 5.53756)
    return (a ** b) * -1


def alpha_82(P):
    """min(rank(decay_linear(delta(open,1.46063),14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume,IndClass.sector),(open*0.634196)+(open*(1-0.634196)),17.4842),6.92131),13.4283))*-1"""
    a = op.rank(op.decay_linear(op.delta(P["open"], 1.46063), 14.8717))
    b = op.ts_rank(op.decay_linear(op.correlation(op.indneutralize(P["volume"], P["ind"]), P["open"], 17.4842), 6.92131), 13.4283)
    return op.ew_min(a, b) * -1


def alpha_87(P):
    """max(rank(decay_linear(delta((close*0.369701)+(vwap*(1-0.369701)),1.91233),2.65461)),Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry),close,13.4132)),4.89768),14.4535))*-1"""
    a = op.rank(op.decay_linear(op.delta(P["close"] * 0.369701 + P["vwap"] * (1 - 0.369701), 1.91233), 2.65461))
    b = op.ts_rank(op.decay_linear(op.abs_(op.correlation(op.indneutralize(data.adv(P, 81), P["ind"]), P["close"], 13.4132)), 4.89768), 14.4535)
    return op.ew_max(a, b) * -1


def alpha_89(P):
    """Ts_Rank(decay_linear(correlation((low*0.967285)+(low*(1-0.967285)),adv10,6.94279),5.51607),3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry),3.48158),10.1466),15.3012)"""
    a = op.ts_rank(op.decay_linear(op.correlation(P["low"], data.adv(P, 10), 6.94279), 5.51607), 3.79744)
    b = op.ts_rank(op.decay_linear(op.delta(op.indneutralize(P["vwap"], P["ind"]), 3.48158), 10.1466), 15.3012)
    return a - b


def alpha_90(P):
    """(rank(close-ts_max(close,4.66719))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry),low,5.38375),3.21856))*-1"""
    a = op.rank(P["close"] - op.ts_max(P["close"], 4.66719))
    b = op.ts_rank(op.correlation(op.indneutralize(data.adv(P, 40), P["ind"]), P["low"], 5.38375), 3.21856)
    return (a ** b) * -1


def alpha_91(P):
    """(Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry),volume,9.74928),16.398),3.83219),4.8667) - rank(decay_linear(correlation(vwap,adv30,4.01303),2.6809)))*-1"""
    a = op.ts_rank(op.decay_linear(op.decay_linear(op.correlation(op.indneutralize(P["close"], P["ind"]), P["volume"], 9.74928), 16.398), 3.83219), 4.8667)
    b = op.rank(op.decay_linear(op.correlation(P["vwap"], data.adv(P, 30), 4.01303), 2.6809))
    return (a - b) * -1


def alpha_93(P):
    """Ts_Rank(decay_linear(correlation(IndNeutralize(vwap,IndClass.industry),adv81,17.4193),19.848),7.54455) / rank(decay_linear(delta((close*0.524434)+(vwap*(1-0.524434)),2.77377),16.2664))"""
    a = op.ts_rank(op.decay_linear(op.correlation(op.indneutralize(P["vwap"], P["ind"]), data.adv(P, 81), 17.4193), 19.848), 7.54455)
    b = op.rank(op.decay_linear(op.delta(P["close"] * 0.524434 + P["vwap"] * (1 - 0.524434), 2.77377), 16.2664))
    return a / b


def alpha_97(P):
    """(rank(decay_linear(delta(IndNeutralize((low*0.721001)+(vwap*(1-0.721001)),IndClass.industry),3.3705),20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871),Ts_Rank(adv60,17.255),4.97547),18.5925),15.7152),6.71659))*-1"""
    a = op.rank(op.decay_linear(op.delta(op.indneutralize(P["low"] * 0.721001 + P["vwap"] * (1 - 0.721001), P["ind"]), 3.3705), 20.4523))
    b = op.ts_rank(op.decay_linear(op.ts_rank(op.correlation(op.ts_rank(P["low"], 7.87871), op.ts_rank(data.adv(P, 60), 17.255), 4.97547), 18.5925), 15.7152), 6.71659)
    return (a - b) * -1


def alpha_100(P):
    """0-(1*((1.5*scale(indneutralize(indneutralize(rank((((close-low)-(high-close))/(high-low))*volume),IndClass.subindustry),IndClass.subindustry))) - scale(indneutralize(correlation(close,rank(adv20),5)-rank(ts_argmin(close,30)),IndClass.subindustry)))*(volume/adv20))"""
    inner = op.rank((((P["close"] - P["low"]) - (P["high"] - P["close"])) / (P["high"] - P["low"])) * P["volume"])
    t1 = 1.5 * op.scale(op.indneutralize(op.indneutralize(inner, P["ind"]), P["ind"]))
    t2 = op.scale(op.indneutralize(op.correlation(P["close"], op.rank(data.adv(P, 20)), 5) - op.rank(op.ts_argmin(P["close"], 30)), P["ind"]))
    return 0 - (1 * ((t1 - t2) * (P["volume"] / data.adv(P, 20))))


ALPHAS = {int(name.split("_")[1]): fn
          for name, fn in list(globals().items())
          if name.startswith("alpha_") and callable(fn)}


def compute_all(P):
    out = {}
    for n, fn in sorted(ALPHAS.items()):
        try:
            out[n] = fn(P)
        except Exception as e:
            print(f"alpha_{n} 失败: {e}", flush=True)
    return out
