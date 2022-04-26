# coding by shiqiu Yang 21372130
# 1900 1 1 was a Monday
def if_leapyear(y):
    """to find if the year is a leap year"""
    if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0:
        return True
    else:
        return False


def days_afterthegivenyear(y):
    """calculate how many days between 1900 and the input year,
    it does not include days in the input year"""
    sumdays = 0
    i = 1900
    while i < y:
        if if_leapyear(i) == True:
            sumdays = sumdays + 366
        else:
            sumdays = sumdays + 365
        i = i + 1
    return sumdays


def days_beforethegivenmouth(m, y):
    """calculate how many days before the input month in the input year"""
    i = 1
    leftdays = 0
    while i < m:
        if i in [1, 3, 5, 7, 8, 10, 12]:
            leftdays = leftdays + 31
        elif i == 2:
            if if_leapyear(y) == True:
                leftdays = leftdays + 29
            else:
                leftdays = leftdays + 28
        else:
            leftdays = leftdays + 30
        i = i + 1
    return leftdays


def checkdate(n):
    """to find how many blanks should be shown in the first line of the calendar"""
    blanks = n % 7
    return blanks


def days_show(m, y):
    """to show us how many days in the month so that we can make the calendar"""
    if m in [1, 3, 5, 7, 8, 10, 12]:
        day = 31
    elif m == 2:
        if if_leapyear(y) == True:
            day = 29
        else:
            day = 28
    else:
        day = 30
    return day


def create_calendar(d,b):
    """to print the calendar!!"""
    ct = 0
    i = 1
    while ct < b:
        print(" \t", end="")#these are bloceks in the first line
        ct = ct + 1
    while i <= d:
        if b != 0:
            if i % 7 != 7 - ct:
                print("{0}\t".format(i), end="")
                i = i + 1
            else:
                print("{0}\t".format(i))
                i = i + 1
        else:
            if i % 7 != 0 :
                print("{0}\t".format(i), end="")
                i = i + 1
            else:
                print("{0}\t".format(i))
                i = i + 1



def final_calendar():
    y = int(input("Please input a year after 1900(include 1900)"))
    m = int(input("Please input the mouth number"))
    d = days_beforethegivenmouth(m,y)+days_afterthegivenyear(y)
    b = checkdate(d)
    day = days_show(m,y)
    print("Mon\tTue\tWed\tThu\tFri\tSat\tSun\t")
    create_calendar(day,b)

if __name__ == "__main__":
    final_calendar()
