#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from svm import SVM
import sys
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    argparse.ArgumentParser(description="Generate a Sequence Vector Machine (SVM).")
    parser.add_argument("--c", help="Set a constant C for the SVM")
    parser.add_argument("--noPlot", help="Do not open a graphical representation of the SVM.", action="store_true")
    parser.add_argument("--printReport", help="Print report of basic SVM data.", action="store_true")
    parser.add_argument("--brute", help="If printing report, calculate LOO error by retraining. Will increase run time significantly.", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    settings = dict()

    if args.noPlot:
        settings['plot'] = False
    else:
        settings['plot'] = True

    if args.printReport:
        settings['report'] = True
    else:
        settings['report'] = False

    if args.brute:
        if args.printReport:
            settings['brute'] = True
        else:
            print("Unable to calculate brute force LOO without report printing enabled.")
            settings['brute'] = False
    else:
        settings['brute'] = False

    if args.c:
        # C Specified
        SVM(C=args.c, plotNow=settings.get('plot'), printReport=settings.get('report'), brute_loo=settings.get('brute'))
    else:
        # Default C
        SVM(plotNow=settings.get('plot'), printReport=settings.get('report'), brute_loo=settings.get('brute'))