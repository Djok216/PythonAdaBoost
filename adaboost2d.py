import matplotlib.pyplot as plt
import sys
from fractions import Fraction


def readData(filePath):
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


def plotPointsWithError(blue, red, errW=None, line=None):
  if not errW:
    n = len(blue[0]) + len(red[0])
    errW = [Fraction(1, n) for _ in range(n)]

  fig, ax = plt.subplots()
  ax.scatter(blue[0], blue[1], c='blue')
  ax.scatter(red[0], red[1], c='red')
  for i, val in enumerate(errW):
    x, y = (blue[0][i], blue[1][i]) if i < len(blue[0]) else (red[0][i - len(blue[0])], red[1][i - len(blue[0])])
    ax.annotate(str(val), (x, y))
  if line:
    if line[0]:
      plt.axvline(3)
    else:
      plt.axhline(3)
  plt.show()


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print "You should run the script with the path to the dataset: python adaboost2d.py path_to_data"
  blue, red = readData(sys.argv[1])