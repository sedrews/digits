prefix = """def pre():
    modal = step([(0,1,1/3),(1,2,1/3),(2,3,1/3)])
    if modal < 1:
        lin = gaussian(30, 9)
    elif modal < 2:
        lin = gaussian(35, 9)
    else:
        lin = gaussian(50, 9)
    ltarget = gaussian(75, 1)
    return lin, ltarget

def D(lin, ltarget):
    h = Hole(0, (0, 10))
    tOn = ltarget + Hole(0, (-10, 0))
    tOff = ltarget + Hole(0, (0, 10))
    isOn = 0
    K = 0.1
    curL = lin
    event("sanity1", tOn < tOff)
    event("sanity2", h > 0)
    event("sanity3", h < 20)"""
body = """
    if isOn > 0.5:
        curL = curL + (h - K * (curL - lin))
        if curL > tOff:
            isOn = 0
    else:
        curL = curL - K * (curL - lin)
        if curL < tOn:
            isOn = 1
"""
body_assert_prefix = "    event(\"body_"
body_assert_suffix = "\", curL < 120)"
suffix1 = """
    Error = curL - ltarget
    if Error < 0:
        Error = Error * -1
    if Error < """
suffix2 = """:
        ret = 0
    else:
        ret = 1
    return ret

def post(Pr):
    thresh = .9
    es = ["sanity1", "sanity2", "sanity3" """
suffix3 = """]
    for e in es:
        if Pr({e : True}) <= thresh:
            return False
    return True
"""

def build(n, t):
    code = prefix
    for i in range(n):
        code += body + body_assert_prefix + str(i) + body_assert_suffix
    code += suffix1 + str(t) + suffix2
    for i in range(n):
        code += ", \"body_" + str(i) + "\""
    code += suffix3
    return code

if __name__ == '__main__':
    print(build(10, 10))
