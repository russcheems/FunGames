"""
Author: Shiqiu Yang
Student number: 21372130
"""


def find_nexthigher(lst):
    """
    The function tries to iterate the list from back to front, find two adjacent ascending numbers, and swap the smaller
    one (i-1,the first of the two) with the smallest number larger than it, Then sort the numbers behind the second
    ascending number(i)
    """
    i = len(lst)-1  # find the last number's index
    while i != 0:  # iterating through list from back to front
        if lst[i] > lst[i-1]:  # find two adjacent numbers that are ascending
            if i+1 <= len(lst)-1 and lst[i-1] < lst[i+1]:  # if there is number that larger than it behind i, it is i+1
                change_number = lst[i-1]  # swap
                lst[i-1] = lst[i+1]
                lst[i+1] = change_number
                lst[i:] = lst[i:][::-1]
                break
            else:  # if there is no number...., then the larger number is i
                change_number = lst[i-1]  # swap
                lst[i-1] = lst[i]
                lst[i] = change_number
                lst[i:] = lst[i:][::-1]
                break
        else:
            i = i - 1
    print(lst)


def find_max(lst):
    while lst != sorted(lst,reverse=True):
        find_nexthigher(lst)


def find_nextlower(lst):
    """
    similar with find_higher
    """
    i = len(lst)-1  # find the last number's index
    while i != 0:  # iterating through list from back to front
        if lst[i] < lst[i-1]:  # find two adjacent numbers that are DESC
            if i+1 <= len(lst)-1 and lst[i-1] > lst[i+1]:  # if there is number that smaller than it behind i, it is i+1
                change_number = lst[i-1]  # swap
                lst[i-1] = lst[i+1]
                lst[i+1] = change_number
                lst[i:] = lst[i:][::-1]
                break
            else:  # if there is no number...., then the smaller number is i
                change_number = lst[i-1]  # swap
                lst[i-1] = lst[i]
                lst[i] = change_number
                lst[i:] = lst[i:][::-1]
                break
        else:
            i = i - 1
    print(lst)


def find_min(lst):
    while lst != sorted(lst):
        find_nextlower(lst)