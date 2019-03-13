import matplotlib.pyplot as plt
import sys
import math
from fractions import Fraction


EPS = 1e-9
def readData(filePath):
  # blue = minus
  # red = plus
  with open(filePath, 'r') as fin:
    n = int(fin.readline())
    blue = [[], []]
    while n > 0:
      n -= 1
      x, y = map(float, fin.readline().split())
      blue[0].append(x)
      blue[1].append(y)
    n = int(fin.readline())
    red = [[], []]
    while n > 0:
      n -= 1
      x, y = map(float, fin.readline().split())
      red[0].append(x)
      red[1].append(y)
  return blue, red


def plotPointsWithError(blue, red, errW, line, coef=1):
  fig, ax = plt.subplots()
  ax.scatter(blue[0], blue[1], c='blue')
  ax.scatter(red[0], red[1], c='red')
  for i, val in enumerate(errW):
    x, y = (blue[0][i], blue[1][i]) if i < len(blue[0]) else (red[0][i - len(blue[0])], red[1][i - len(blue[0])])
    ax.annotate(str(Fraction(val).limit_denominator(10000)), (x, y))
  div = ['+/-', '-/+'][coef == 1]
  plt.text(1.01, 1.06, div, fontsize=20, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5))
  if line[0]:
    plt.axvline(line[1])
  else:
    plt.axhline(line[1])
  plt.show()


def getAllDecisionStumps(blue, red):
  def check(val, s):
    if val in s:
      return True
    for x in s:
      if abs(x - val) < EPS:
        return True
    return False
  lines = set()
  pblue = list(zip(blue[0], blue[1]))
  pred = list(zip(red[0], red[1]))
  xblue = set(x[0] for x in pblue)
  xred = set(x[0] for x in pred)
  for xb in xblue:
    for xr in xred:
      if xb != xr:
        val = (xb + xr) * 0.5
        if not check(val, xblue) and not check(val, xred):
          lines.add((True, val))
  yblue = set(x[1] for x in pblue)
  yred = set(x[1] for x in pred)
  for yb in yblue:
    for yr in yred:
      if yb != yr:
        val = (yb + yr) * 0.5
        if not check(val, yblue) and not check(val, yred):
          lines.add((False, val))
  lines.add((True, min(min(xblue), min(xred)) - 1))
  lines.add((False, min(min(yblue), min(yred)) - 1))
  return list(lines)


def pickStump(stumps, blue, red, errW):
  ans, err, coef = stumps[0], 0.5, 1
  for stump in stumps:
    currErr, currCoef = 0.0, 1
    if stump[0]:
      for i, x in enumerate(blue[0]):
        if x > stump[1]:
          currErr += errW[i]
      for i, x in enumerate(red[0]):
        if x <= stump[1]:
          currErr += errW[i + len(blue[0])]
    else:
      for i, y in enumerate(blue[1]):
        if y > stump[1]:
          currErr += errW[i]
      for i, y in enumerate(red[1]):
        if y <= stump[1]:
          currErr += errW[i + len(blue[0])]
    if currErr == 0.5:
      continue
    if currErr > 0.5:
      currErr = 1 - currErr
      currCoef = -1
    if currErr < err:
      err, coef, ans = currErr, currCoef, stump
  return coef, ans, err


def adaBoost2d(blue, red, maxIter=500, withPlots=False):
  n = len(blue[0]) + len(red[0])
  errW = [Fraction(1, n) for _ in range(n)]
  decisionStumps = getAllDecisionStumps(blue, red)
  alfa, stumps, errHistory = [], [], []
  vals = [0] * n
  while maxIter > 0:
    maxIter -= 1
    coef, stump, err = pickStump(decisionStumps, blue, red, errW)
    if withPlots:
      plotPointsWithError(blue, red, errW, stump, coef)
    stumps.append(stump)
    errHistory.append(err)
    if abs(err - 0.5) < EPS:
      break
    alfa.append(0.5 * math.log((1 - err) / err))
    toStop = True
    if stump[0]:
      for i, x in enumerate(blue[0]):
        if coef * x > coef * stump[1]:
          errW[i] /= Fraction(2 * err)
          vals[i] += alfa[-1]
        else:
          errW[i] /= Fraction(2 * (1 - err))
          vals[i] -= alfa[-1]
        if vals[i] > 0:
          toStop = False
      for i, x in enumerate(red[0]):
        if coef * x <= coef * stump[1]:
          errW[i + len(blue[0])] /= Fraction(2 * err)
          vals[i + len(blue[0])] -= alfa[-1]
        else:
          errW[i + len(blue[0])] /= Fraction(2 * (1 - err))
          vals[i + len(blue[0])] += alfa[-1]
        if vals[i + len(blue[0])] < 0:
          toStop = False
    else:
      for i, y in enumerate(blue[1]):
        if coef * y > coef * stump[1]:
          errW[i] /= Fraction(2 * err)
          vals[i] += alfa[-1]
        else:
          errW[i] /= Fraction(2 * (1 - err))
          vals[i] -= alfa[-1]
        if vals[i] > 0:
          toStop = False
      for i, y in enumerate(red[1]):
        if coef * y <= coef * stump[1]:
          errW[i + len(blue[0])] /= Fraction(2 * err)
          vals[i + len(blue[0])] -= alfa[-1]
        else:
          errW[i + len(blue[0])] /= Fraction(2 * (1 - err))
          vals[i + len(blue[0])] += alfa[-1]
        if vals[i + len(blue[0])] < 0:
          toStop = False
    if toStop:
      break
  print("ALFA: ", ' '.join(map(str, alfa)))
  print("H: ", ' '.join('-' if x < 0 else '+' for x in vals))
  plt.plot(errHistory)
  plt.show()

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("You should run the script with the path to the dataset: python adaboost2d.py path_to_data")
    exit(0)
  blue, red = readData(sys.argv[1])
  adaBoost2d(blue, red, withPlots=len(sys.argv) > 2)