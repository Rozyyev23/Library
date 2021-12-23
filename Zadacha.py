# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 00:10:04 2021

@author: Windows 10 Pro
"""
from __future__ import print_function, division
import numpy as np  # для работы с массивами
import matplotlib.pyplot as plt  # для работы с графиками
import pandas as pd  # для создания датафрейма
import math  # для использвания тригонометрических функций
from sympy import diff, symbols, cos, sin
from datetime import datetime  # измерение длительности выполнения программы
from scipy import integrate  # для нахождения интеграла
import numpy as np
from datetime import datetime
import operator
import matplotlib.pyplot as plt
import csv
import copy
from fractions import Fraction
from numpy import *
from scipy import *
from math import log, exp
from numpy import array, dot, linalg, arange
from random import randint
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy import spatial
from scipy import optimize
import seaborn as sns
from sympy import symbols, expand, lambdify, sqrt, cos, sin, log, exp

import numpy.random as rn
import matplotlib as mpl
from scipy.optimize import leastsq, curve_fit
from sklearn.metrics import mean_squared_error
from scipy.optimize import leastsq
from random import uniform, randint
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d


def __zadacha1__():
    '''Дифференциирование с пределом и шагом'''
    start = float(input('Введите нижний предел:'))
    end = float(input('Введите верхний предел:'))
    step = float(input('Введите шаг дифференцирования:'))
    start_time = datetime.now()

    x_axis = np.arange(start, end+step, step).tolist()  # создание массива х
    x_axis = [round(v, 1) for v in x_axis]  # округление элементов массива

    def funct(x):
        try:  # танцы с бубном или как подорожник лечит всё
            y = math.sin(x)/x
        except ZeroDivisionError:
            print('Точка разрыва:', x)
            return x
        else:
            return y

    def derivative(a):
        b = (funct(a+step)-funct(a-step))/(2*step)  # функция нахождения f'
        return b
    # создание списков, для дальнейшего добавления данных в них
    y_axis = []
    y_derivative = []

    for i in x_axis:  # цикл для добавления всех данных массивов в списки
        y_axis.append(funct(i))
        y_derivative.append(derivative(i))
    # создание датафрейма
    data = pd.DataFrame({'x': x_axis,
                         'y': y_axis,
                         'производная': y_derivative})
    data.to_excel('./derivative_table.xlsx')  # запись датафрейма в эксель

    end_time = datetime.now()
    print('Длительность выполнения: {}'.format(end_time - start_time))
    data  # для наглядного примера
    df = pd.read_excel("derivative_table.xlsx")  # чтение файла
    # задаем размер нашего графика, который будет отображаться на экране
    plt.figure(figsize=(8, 5))
    df.plot(x='x', y='y', c='black')  # оси и цвет графика
    # название графика
    plt.title('График функции f(x)', fontsize=15, color='orange')
    plt.xlabel('Ось Х', color='blue')  # название осей
    plt.ylabel('Ось Y', color='blue')
    plt.legend(['f(x)'])  # легенда графика
    plt.grid(True)  # сетка
    # абсолютно схожие строки, что и в предыдущем блоке
    df = pd.read_excel("derivative_table.xlsx")
    plt.figure(figsize=(8, 5))
    df.plot(x='x', y='производная', c='black')
    plt.title("График функции f'(x)", fontsize=15, color='orange')
    plt.xlabel('Ось Х', color='blue')
    plt.ylabel('Ось Y', color='blue')
    plt.legend(["f'(x)"])
    plt.grid(True)



def __zadacha1_2__():
    '''Численное интегрирование'''
    start_time = datetime.now()

    # функция для нахождения численного интеграла методом трапеции
    def trap(f, a, b, step_i):
        numb_of_steps = (b - a) / step_i
        print('Количество трапеций:', numb_of_steps)
        s = (f(a) + f(b)) / 2
        x = a + step_i
        while (x <= b - step_i):
            s += f(x)
            x += step_i
        return step_i * s

    def test():
        # функция для ввода данных(функция,нижний предел, верхний предел, шаг)
        print(trap(lambda x: np.sin(x ** 2) / x ** 2, -2, 2, 0.165))
    test()

    end_time = datetime.now()
    print('Длительность выполнения: {}'.format(end_time - start_time))
    plt.figure(figsize=(8, 5))
    y = lambda x: np.sin(x ** 2) * x ** 2
    y2 = lambda x: 2 * x ** 3 * cos(x ** 2) + 2 * x * sin(x ** 2)
    xs = np.linspace(-10, 10, 100)
    xr = np.linspace(-10, 10, 100)
    plt.title("sin(x**2)*x**2, f'(x)", fontsize=17)
    plt.xlabel('ось X', fontsize=15, color='black')
    plt.ylabel('ось Y', fontsize=15, color='black')
    plt.plot(xs, [y(x) for x in xs], color='red', label='sin(x**2)*x**2')
    plt.plot(xr, [y2(x) for x in xr],
             color='black', label="2*x**3*cos(x**2) + 2*x*sin(x**2)")
    plt.legend()
    plt.grid()
    plt.show()
    
    
def __zadacha2__():
    ''' Операции с матрицами'''
    x = int(input('Введите количество строк: '))
    y = int(input('Введите количество столбцов: '))
    Matrix = []
    for i in range(x):
        Matrix.append([])
        for j in range(y):
            ''' Если захотите ввести матрицу с комплексными \
                числами или числа типа
            плавающей точкой поменяйте внизу int на complex или float'''
            Matrix[i] += [int(input())]  # <-------Здесь
            print(Matrix)

    ''' фиксируем время начала работы (и конца работы). далее создаем новый
    список функцией zip и инвертированными (с помощью *)
    значениями исходной матрицы.'''
    start_time = datetime.now()
    trans_matrix_1 = [list(x) for x in zip(*Matrix)]
    print('Транспонированная:', '\n', trans_matrix_1)
    end_time = datetime.now()
    print('Длительность выполнения: {}'.format(end_time - start_time))

    ''' для операций с матрицами создаем класс "матрица" с \
        несколькими методами
    используем библиотеку operator для работы с элементами матрицы'''
    start_time = datetime.now()

    class Matrix:
        def __init__(self, lst):
            self.lst = lst

        def __str__(self):
            lst_of_strs = ['\t'.join(map(str, row)) for row in self.lst]
            return '\n'.join(lst_of_strs)

        def size(self):
            return len(self.lst), len(self.lst[0])

        def operation_on_pairs(self, operand_2, op):
            if self.size() == operand_2.size():
                return [[op(a, b) for a, b in zip(row_1, row_2)]
                        for row_1, row_2 in zip(self.lst, operand_2.lst)]

        def __add__(self, operand_2):  # магический метод сложения
            return Matrix(self.operation_on_pairs(operand_2, operator.add))

        def __sub__(self, operand_2):  # магический метод вычитания
            return Matrix(self.operation_on_pairs(operand_2, operator.sub))

        def __mul__(self, operand_2):  # магический метод умножения
            def mul(row, col):
                return sum(a * b for a, b in zip(row, col))

            res_mtrx = Matrix([])
            for row in self.lst:
                if isinstance(operand_2, int):
                    res_mtrx.lst.append([col * operand_2 for col in row])
                else:
                    res_mtrx.lst.append([mul(row, col)
                                        for col in zip(*operand_2.lst)])
            return res_mtrx

        __rmul__ = __mul__

    # ЗАДАЕМ МАТРИЦЫ
    m = [[1, 1, 0], [0, 2, 10], [10, 15, 30]]
    n = [[2, 1, 2], [3, 2, 5], [10, 15, 30]]
    c = [[2, 1, 2], [3, 2, 5], [10, 15, 30]]
    m = Matrix(m)
    n = Matrix(n)
    с = Matrix(c)

    print(n + m - с)  # СЮДА ПИШЕМ ПРЕОБРАЗОВАНИЯ С МАТРИЦАМИ

    end_time = datetime.now()
    print('Длительность выполнения: {}'.format(end_time - start_time))
    
    
def can_be_int(chislo):
    '''Название: can_be_int
       Входные параметры (+формат): chislo - строка 
       Выходные параметры: да/нет (bool)
       Краткое описание: проверка, является ли элемент \
           действительным числом'''
    try:
        a = int(chislo)
        return True
    except:
        return False
    
def vvod_s_klav():
    '''Название: vvod_s_klav
       Входные параметры (+формат):  
       Выходные параметры:  matrix - матрица (список списков)
       Краткое описание: пользователь вводит размерность, мы \
           проверяем ее на натуральность, затем просим заполнить саму \
               матрицу исходя из размерности'''
    raz = input('Введите размерность наших \
                квадратных матриц (одно натур.число): ')
    try:
        raz = int(raz)
        while int(raz) <= 0:
            raz = input('Невозможная размерность. Введите заново: ')
        raz = int(raz)
    except:
        print('Число должно быть натуральным')
        raz = input('Невозможная размерность. Введите заново: ')
        while can_be_int(raz) == False or int(raz) < 0:
            raz = input('Невозможная размерность. Введите заново: ')
        raz = int(raz)
    print(f'Размерность нашей матрицы: {raz}')
        
    

    matrix = []
    for i in range(raz):
            stroka = input(f'Введите {i+1} строку\
                           (все элементы через пробел): ')
            new_stroka2 = ''
            if len(stroka.split(' ')) == raz+1:

                if 'i' in stroka:
                    #Заменяем все i на j
                    new_stroka = ''.join(['j' if el == 'i' else el for\
                                          el in stroka])
                    #Сначала реальная, потом мнимая часть
                    for element in new_stroka.split(' '):
                        if 'j' in element:
                            if element[-1] == 'j':
                                pass
                            else:
                                for ind,symbol in enumerate([i for i in\
                                                             element]):
                                    if symbol == '+' and ind != 0:
                                        p1 = ''.join([el  for el\
                                                      in element.split('+') \
                                                          if 'j' not in el])
                                        if element[0] == '-':
                                            p2 = ''.join([el  for el in \
                                                          element.split('+') \
                                                              if 'j' in el])
                                        else:
                                            p2 = '+' +\
                                                ''.join([el  for el in \
                                                         element.split('+') if\
                                                             'j' in el])
                                    elif symbol == '-' and ind != 0:
                                        stroka = element[:ind] + 'тут' +\
                                            element[ind+1]
                                        p1 = '-' + ''.join([el  for el in \
                                                            stroka.split('тут')\
                                                                if 'j' \
                                                                    not in el])
                                        if stroka[0] == '-':
                                            p2 = ''.join([el for el in \
                                                          stroka.split('тут') \
                                                              if 'j' in el])
                                        else:
                                            p2 = '+' +\
                                                ''.join([el for el in\
                                                         stroka.split('тут') \
                                                             if 'j' in el])
                                element = f'{p1}{p2}'  
                            new_stroka2 += element
                            
                        else:
                            new_stroka2 += element
                        new_stroka2 += ' '
                
                    
                else:
                    new_stroka2 = stroka
                
                matrix.append([complex(el) for el in\
                               new_stroka2.split(' ') if el!=''])
            else:
                while len(stroka.split(' ')) != raz+1:
                    print('Введенное кол-во элементов не соответсвует \
                          заявленной размерности. Повторите ввод.')
                    stroka = input(f'Введите {i+1} строку \
                                   (все элементы через пробел): ')

                if 'i' in stroka:
                    #Заменяем все i на j
                    new_stroka = ''.join(['j' if el == 'i' else el for el in\
                                          stroka])  #new_stroka - строка без i
                    #Сначала реальная, потом мнимая часть
                    for element in new_stroka.split(' '):
                        if 'j' in element:
                            if element[-1] == 'j':
                                pass
                            else:
                                for ind,symbol in enumerate([i for \
                                                             i in element]):
                                    if symbol == '+' and ind != 0:
                                        p1 = ''.join([el  for el in\
                                                      element.split('+') \
                                                          if 'j' not in el])
                                        if element[0] == '-':
                                            p2 = ''.join([el  for el in\
                                                          element.split('+') \
                                                              if 'j' in el])
                                        else:
                                            p2 = '+' +\
                                                ''.join([el  for el in\
                                                         element.split('+')\
                                                             if 'j' in el])
                                    elif symbol == '-' and ind != 0:
                                        stroka = element[:ind] + 'тут' +\
                                            element[ind+1]
                                        p1 = '-' +\
                                            ''.join([el  for el in\
                                                     stroka.split('тут') if\
                                                         'j' not in el])
                                        if stroka[0] == '-':
                                            p2 = ''.join([el  for el in\
                                                          stroka.split('тут')\
                                                              if 'j' in el])
                                        else:
                                            p2 = '+'+\
                                                ''.join([el  for el in \
                                                         stroka.split('тут') \
                                                             if 'j' in el])
                                element = f'{p1}{p2}'  
                            new_stroka2 += element
                            
                        else:
                            new_stroka2 += element
                        new_stroka2 += ' '
                
                    
                else:
                    new_stroka2 = stroka
                
                matrix.append([complex(el) for el in \
                               new_stroka2.split(' ') if el!=''])

    print('Вы ввели матрицу: ', *matrix, sep='\n')
    return matrix


def proverka(lst_x, lst_x0, tochn):

    for i in range(len(lst_x)):
        if abs(lst_x[i] - lst_x0[i]) > tochn:
            return False
    return True 


def reshit_yiakobi(matrix, tochn):
    lst_x0 = []
    lst_x = []
    
    for i in range(len(matrix)):
        lst_x0.append(complex(0,0))
        
    #Преобразование:
    for i in range(len(matrix)):
        tecysh = matrix[i][i]
        matrix[i][i] = complex(0,0)
        for ind, el in enumerate(matrix[i]):

            matrix[i][ind] = (-1)*el/tecysh
       
    #Расширение единичной:
#     for i in range(len(matrix)):
#         for j in range(len(matrix)):
#             if i == j:
#                 matrix[i].append(1)
#             else:
#                 matrix[i].append(0)
                
#     print(*matrix, sep='\n') 
    count = 0
    while True:
        count += 1
        for i in range(len(matrix)):
            lst_x.append(sum([matrix[i][j] * lst_x0[j] for j in\
                              range(len(matrix))]) + matrix[i][len(matrix)])
        if proverka(lst_x, lst_x0, tochn) == False and count <= 100:
            for ind, el in enumerate(lst_x0):
                
                lst_x0[ind] = lst_x[ind]
                
            lst_x = []
        else:
            break
#     print(f'Решение методом Якоби: {lst_x}')
    if count > 101:
        return 0
    else:
        return lst_x
    
    
def reshit_gauss(matrix, okrug):
    
    #Расширение единичной:
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                matrix[i].append(complex(1,0))
            else:
                matrix[i].append(complex(0,0))
#     print(f'Матрица после расширения: {matrix}')
    
    for i in range(len(matrix)):
        tot_samyi = matrix[i][i]
        for ind, el in enumerate(matrix[i]):
            
            matrix[i][ind] = el/tot_samyi
        
        for j in range(len(matrix)):
            if j != i:
                tot_samyi2 = matrix[j][i]
                for ind, el in enumerate(matrix[j]):
                    matrix[j][ind] = el - matrix[i][ind]*tot_samyi2
        
    print(*matrix, sep='\n')
    rezult = [stroka[len(matrix)] for stroka in matrix]
    ob_gauss = []
    for stroka in matrix:
        ob_gauss.append(stroka[len(matrix)+1::])
    return(rezult, ob_gauss)


def __zadacha3_1__():
    def preobrazovanie(matrix):
        for i in range(len(matrix)):
            if matrix[i][i] == 0:
                for j in range(len(matrix)):
                    if matrix[j][i] != 0:
                        matrix[i] = list(map(sum, zip(matrix[i], matrix[j])))
                        break
        return matrix
    
    def obyslovlennost(matrix, obrat_matrix):

        def norma(matrix):
            norm = 0
            for i in range(len(matrix)):
                if sum([abs(el) for el in matrix[i]]) > 0:
                    norm = sum([abs(el) for el in matrix[i]])
            return norm

    #    print(f'Число обуслов. {norma(matrix)*norma(obrat_matrix)}')
        return norma(matrix)*norma(obrat_matrix)

    def get_cofactor(matrix, i, j):
        return [row[: j] + row[j+1:] for row in (matrix[: i] + matrix[i+1:])]
    def is_singular(matrix):
        n = len(matrix)
        if (n == 2):
            val = matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1]
            return val

        det = 0
        for i in range(n):
            s = (-1)**i
            sub_det = is_singular(get_cofactor(matrix, 0, i))
            det += (s*matrix[0][i]*sub_det)
        return det

    def reshit_dan_matrix(tochn = 0.001):
    #     matrix = vvod_random()
        matrix = vvod_s_klav()
    #    matrix = vvod_csv()

        #Проверяем, чтобы на диагонали не было нулей
        fl = 0
        for i in range(len(matrix)):
            if matrix[i][i] == 0:
                fl += 1
                break
        if fl == 1:
            matrix = preobrazovanie(matrix)
            print('Преобразуем матрицу:')
            print(*matrix, sep = '\n')



        kopia_for_gauss = copy.deepcopy(matrix)
        kopia_for_gauss_2 = copy.deepcopy(matrix)
        for ind, stroka in enumerate(kopia_for_gauss):
            kopia_for_gauss[ind] = stroka[:-1]
        #Проверем на вырожденность
        if is_singular([el[:-1] for el in matrix]) == '0j' or\
            is_singular([el[:-1] for el in matrix]) == '0' or \
                is_singular([el[:-1] for el in matrix]) == 0:
            print('Вы ввели вырожденную матрицу')
        else: 
            M_for_np = [[el for el in stroka] for stroka in matrix]
            M = np.array([el[:-1] for el in M_for_np])
            M_inv = np.linalg.inv(M)
    #         print(f'M {M}')
    #         print(f'M_inv {M_inv}')
            rez_ya = reshit_yiakobi(matrix, tochn)

            print('Результат методом Якоби (X): ')
            if rez_ya != 0:
                to_print = []
                to_print.append([f'{el}' for el in rez_ya])
                print(*to_print, sep='\n')
    #             print(*rez_ya, sep = '\n')
                print(f'Число обусловленности (Cord):\
                      {obyslovlennost(M, M_inv)}')
            else:
                print('Матрица расходится')
                print(f'Число обусловленности: {obyslovlennost(M, M_inv)}')
            print(f'Обратная матрица (A^-1):')
            print(*M_inv, sep = '\n')



            if rez_ya != 0 and obyslovlennost(M, M_inv) < 100:
                return
            else:
                if rez_ya == 0:
                    print('------------------------------------------------\
                          --------------------------------')
                    print('Считаем методо Ж-Гаусса, так как кол-во итераций\
                          методом Якоби превысило 100')
                chislo = len(str(tochn))-2      

                rez_gauss, obrat_gauss = reshit_gauss(kopia_for_gauss_2,\
                                                      chislo)


                if obyslovlennost(kopia_for_gauss, obrat_gauss) != 0:
                    print('--------------------------------------------------\
                          ------------------------------')
                    print(f'Результат методом Ж.Гаусса (X):')
                    to_print = []
                    to_print.append([f'{el}' for el in rez_gauss])
                    print(*to_print, sep='\n')

                    print(f'Обратная матрица методом Ж.Гаусса (A^-1):')
                    to_print = []
                    for stroka in obrat_gauss:
                        to_print.append([f'{el}' for el in stroka])
                    print(*to_print, sep='\n')
                    print('Число обусловленности (Cord): ',\
                          obyslovlennost(kopia_for_gauss, obrat_gauss))
                else:
                    print('Что-то не так')

    reshit_dan_matrix(tochn = 0.01)
    
    
def preobrazovanie(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] == 0:
            for j in range(len(matrix)):
                if matrix[j][i] != 0:
                    matrix[i] = list(map(sum, zip(matrix[i], matrix[j])))
                    break
    return matrix


def reshit_gauss(matrix):
    
    #Проверяем, чтобы на диагонали не было нулей
    fl = 0
    for i in range(len(matrix)):
        if matrix[i][i] == 0:
            fl += 1
            break
    if fl == 1:
        matrix = preobrazovanie(matrix)
#         print('Преобразуем матрицу:')
#         print(*matrix, sep = '\n')
    
    for i in range(len(matrix)):
        tot_samyi = matrix[i][i]
        for ind, el in enumerate(matrix[i]):
            
            matrix[i][ind] = el/tot_samyi
        
        for j in range(len(matrix)):
            if j != i:
                tot_samyi2 = matrix[j][i]
                for ind, el in enumerate(matrix[j]):
                    matrix[j][ind] = el - matrix[i][ind]*tot_samyi2
        
#     print(*matrix, sep='\n')
    rezult = [stroka[len(matrix)] for stroka in matrix]
    
    return rezult


def __zadacha4_1__():
    '''Интеполяция методом Лагранжа'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

    return lst_csv

    def poisk_coeff(lst_x, lst_y):
        '''Название: poisk_coeff
           Входные параметры: lst_x - список х координат точек,
           lst_y - список у координат точек
           Выходные параметры:
           stroka_lagr - строка - интерполяционный многочлен Лагранжа
           Краткое описание:
           Создаем матрицу для нахождения коэфф в многочлене Л.,
           выводим готовый многочлен'''
        matrix = []
        for i in range(len(lst_x)):
            matrix.append([lst_x[i] ** n for n in range(len(lst_x))] +\
                          [lst_y[i]])

        lst_coeff = reshit_gauss(matrix)[::-1]   # от 0 степени до 2
        # print(lst_coeff)
        stroka_lagr = 'Интерполяционный многочлен Лагранжа: y = '
        for i in range(len(lst_coeff) - 1):
            if lst_coeff[i] != 0:
                stroka_lagr += f'({lst_coeff[i]})*' + f'x**{len(lst_coeff) - i - 1} +'
        if lst_coeff[-1] != 0:
            stroka_lagr += f'({lst_coeff[-1]})'
        else:
            stroka_lagr = stroka_lagr[:-1]

        return stroka_lagr

    def lst_lagranz(lst_x, lst_y, stroka):
        lst_rez = []
        for i in range(len(lst_x)):
            x = lst_x[i]
            lst_rez.append([lst_x[i], lst_y[i], eval(stroka)])
        return lst_rez

    def print_interpol_lagranz(lst, stroka):
        lst_x = np.array([el[0] for el in lst], dtype=float)
        lst_y = np.array([el[1] for el in lst], dtype=float)

        lst_x_new = np.linspace(np.min(lst_x), np.max(lst_x), 100)
        global lst_for_rez
        lst_for_rez = []

        def lagranz(x, y, t):
            z = 0
            for j in range(len(y)):
                p1 = 1
                p2 = 1
                for i in range(len(x)):
                    if i == j:
                        p1 = p1 * 1
                        p2 = p2 * 1
                    else:
                        p1 = p1 * (t-x[i])
                        p2 = p2 * (x[j]-x[i])
                z = z + y[j] * p1 / p2
                lst_for_rez.append(z)
            return z
        lst_y_new = [lagranz(lst_x, lst_y, i) for i in lst_x_new]
        # lst_y_new = [eval(stroka[41:]) for x in lst_x_new]

        plt.plot(lst_x, lst_y, 'o', lst_x_new, lst_y_new)
        plt.legend(['data', 'интерпол. фун.'], loc='best')
        plt.grid(True)
        plt.title(f"Интерполяция методом Лагранжа.")
        plt.show()
        print(stroka)

    def interpol_meth_lag(lst, print_all=True):
        stroka = poisk_coeff([el[0] for el in lst], [el[1] for el in lst])
        rezult = lst_lagranz([el[0] for el in lst], [el[1] for el in lst],\
                             stroka[41:])
        if print_all:
            print(*rezult, sep='\n')
            # print(stroka)
        if print_all:
            print_interpol_lagranz(lst, stroka)
        return stroka[41:]
    interpol_meth_lag(vvod_csv())
    
    
def __zadacha4_2__():
    '''Интерполяция методом Ньютона'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv    

    def factor(n):
        r = 1
        for i in range(1, n + 1):
            r = r * i
        return r

    def lst_newton(lst_x, lst_y, stroka):
        lst_rez = []
        for i in range(len(lst_x)):
            x = lst_x[i]
            lst_rez.append([lst_x[i], lst_y[i], eval(stroka)])
        return lst_rez

    def print_newton(lst_x, lst_y, stroka):
        lst_x_new = np.linspace(np.min(lst_x), np.max(lst_x), 100)

        lst_y_new = [eval(stroka) for x in lst_x_new]
    #     print(lst_for_rez)

        plt.plot(lst_x, lst_y, 'o', lst_x_new, lst_y_new)
        plt.legend(['data', 'интерпол. фун.'], loc='best')
        plt.grid(True)
        plt.title(f"Интерполяция методом Ньютона (вперед):  ")
        plt.show()
        print(f"y = {stroka}")

    def print_newton_naz(lst_x, lst_y, stroka):
        lst_x_new = np.linspace(np.min(lst_x), np.max(lst_x), 100)

        lst_y_new = [eval(stroka) for x in lst_x_new]
    #     print(lst_for_rez)

        plt.plot(lst_x, lst_y, 'o', lst_x_new, lst_y_new)
        plt.legend(['data', 'интерпол. фун.'], loc='best')
        plt.grid(True)
        plt.title(f"Интерполяция методом Ньютона (назад) ")
        plt.show()
        print(f"y = {stroka}")

    def first_int_newton(lst, print_all=True):  # ДЛЯ РАВНООТСТОЯЩИХ
        lst_x = [el[0] for el in lst]
        lst_y = [el[1] for el in lst]

        # находим значение h
        h = lst_x[1] - lst_x[0]

        # находим конечные разности
        r = 1
        lst_raznosty = [lst_y]
        for i in range(len(lst_x)-1):
            l = [-lst_raznosty[r - 1][j] +\
                 lst_raznosty[r - 1][j + 1] for j in range(len(lst_x)-r)]
            lst_raznosty.append(l)
            r += 1
    #     print(f'Конечные разности: {lst_raznosty}')

        # составляем наш многочлен
        stroka_newton = f'{lst_raznosty[0][0]} +'
        for i in range(1, len(lst_raznosty)):
            a = lst_raznosty[i][0] / (factor(i) * (h ** i))
            p = ''
            for j in range(i):
                p += f'*(x - x{j})'
            if a != 0:
                stroka_newton += f'{a}{p} + '
        stroka_newton = stroka_newton[:-2]

        # переделываем в порядке возрастания степеней
        l = stroka_newton.split('+')
        stroka_newton2 = ''
        for el in l[::-1]:
            stroka_newton2 += f'{el} + '
        stroka_newton2 = stroka_newton2[:-2]
        if print_all:
            print(f'Интерполяционный многочлен Ньютона(до подстановки):\
                  y = {stroka_newton2}')

        # Канононичная форма(?)
        stroka_newton3 = stroka_newton2
        dict_x = {}
        for i in range(len(lst_x)):
            dict_x[f'x{i}'] = lst_x[i]

        for key, v in dict_x.items():
            stroka_newton3 = stroka_newton3.replace(f'{key}', f'{v}')
        if print_all:
            print(f'Интерполяционный многочлен Ньютона(после подстановки):\
                  y = {stroka_newton3}')

        # заполняем список
        lst_n = lst_newton(lst_x, lst_y, stroka_newton3)
        if print_all:
            print(*lst_n, sep='\n')

        if print_all:
            print_newton(lst_x, lst_y, stroka_newton3)
        return stroka_newton3

    first_int_newton(vvod_csv())

    def second_int_newton(lst, print_all=True):  # ДЛЯ РАВНООТСТОЯЩИХ
        lst_x = [el[0] for el in lst]
        lst_y = [el[1] for el in lst]

        # находим значение h
        h = lst_x[1] - lst_x[0]

        # находим конечные разности
        r = 1
        lst_raznosty = [lst_y]
        for i in range(len(lst_x) - 1):
            l = [-lst_raznosty[r - 1][j] \
                 + lst_raznosty[r - 1][j + 1] for j in range(len(lst_x)-r)]
            lst_raznosty.append(l)
            r += 1
    #     print(f'Конечные разности: {lst_raznosty}')

        # составляем наш многочлен
        stroka_newton = f'{lst_raznosty[0][-1]} +'
        for i in range(1, len(lst_raznosty)):

            a = lst_raznosty[i][-1] / (factor(i) * (h ** i))
            p = ''
            for j in range(i):
                p += f'*(x - x{len(lst_raznosty) - j - 1})'
            if a != 0:
                stroka_newton += f'{a}{p} + '
        stroka_newton = stroka_newton[:-2]

        # переделываем в порядке возрастания степеней
        l = stroka_newton.split('+')
        stroka_newton2 = ''
        for el in l[::-1]:
            stroka_newton2 += f'{el} + '
        stroka_newton2 = stroka_newton2[:-2]
        if print_all:
            print(f'Интерполяционный многочлен Ньютона "назад"\
                  (до подстановки): \n y = {stroka_newton2}')

        # Канононичная форма(?)
        stroka_newton3 = stroka_newton2
        dict_x = {}
        for i in range(len(lst_x)):
            dict_x[f'x{i}'] = lst_x[i]

        for key, v in dict_x.items():
            stroka_newton3 = stroka_newton3.replace(f'{key}', f'{v}')
        if print_all:
            print(f'Интерполяционный многочлен Ньютона "назад"\
                  (после подстановки): \n y = {stroka_newton3}')

        # заполняем список
        lst_n = lst_newton(lst_x, lst_y, stroka_newton3)
        if print_all:
            print(*lst_n, sep='\n')
        if print_all:
            print_newton_naz(lst_x, lst_y, stroka_newton3)
        print(stroka_newton3)
        return stroka_newton3

    second_int_newton(vvod_csv())
    
    
def __zadacha4_3__():
    '''Интерполяция методом кубического сплайна'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv

    def lst_kub(lst_x, lst_y, lst_stroka):
        lst_rez = []
        for i in range(len(lst_x) - 1):
            x = lst_x[i]
            lst_rez.append([lst_x[i], lst_y[i], eval(lst_stroka[i])])
        return lst_rez

    def print_kub(lst_x, lst_y, lst_stroka):
        lst_x_new = np.linspace(np.min(lst_x), np.max(lst_x), 100)

        index = -1
        lst_y_new = []
        for i, x in enumerate(lst_x_new):
            if x >= lst_x[index]:
                index += 1
            lst_y_new.append(eval(lst_stroka[index]))

        plt.plot(lst_x, lst_y, 'o', lst_x_new, lst_y_new)
        plt.grid(True)
        plt.title(f"Интерполяция методом Кубического сплайна: ")
        plt.show()

    def kub_sp(lst, print_all=True):  # ДЛЯ РАВНООТСТОЯЩИХ
        lst_x = [el[0] for el in lst]
        lst_y = [el[1] for el in lst]

        # Находим h:
        h = lst_x[1] - lst_x[0]

        # Вычисляем коэффициенты ci
        lst_c = [0]  # т.к. c1 = 0
        # Создаем нашу матрицу
        matrix = []
        for i in range(len(lst) - 2):
            stroka = []
            for j in range(len(lst) - 2):
                if i == j:
                    stroka.append(4 * h)
                elif abs(j-i) == 1:
                    stroka.append(h)
                else:
                    stroka.append(0)
            matrix.append(stroka)
        for i in range(len(matrix)):
            matrix[i].append(3 / h * (lst_y[i + 2] - 2 * lst_y[i + 1] +\
                                      lst_y[i]))

        lst_c += reshit_gauss(matrix) + [0, 0]
    #     print(lst_c)

        # Находим все di
        lst_d = []
        for i in range(len(lst) - 1):
            lst_d.append((lst_c[i + 1] - lst_c[i]) / 3 * h)
        lst_d += [0]
    #     print(lst_d)

        # Находим все bi
        lst_b = []
        for i in range(1, len(lst)):
            lst_b.append((lst_y[i] - lst_y[i - 1]) / h - (lst_c[i + 1] + 2 *\
                                                          lst_c[i]) * h / 3)
    #     print(lst_b)

        # Находим все ai
        lst_a = lst_y[:-1]
    #     print(lst_a)

        # Выводи все формулы, входящие в сплайн
        lst_spline = []
        for i in range(len(lst) - 1):
            lst_spline.append(f'{lst_a[i]} + ({lst_b[i]})*(x-{lst_x[i]}) +\
                              ({lst_c[i]})*(x-{lst_x[i]})**2 +\
                                  ({lst_d[i]})*(x-{lst_x[i]})**3')
        for i in range(len(lst_spline)):
            print(f'На интервале \
                  [{lst_x[i]},{lst_x[i+1]}]: y = {lst_spline[i]}')

        # Составляем массив [xi, yi, fi]
        lst_k = lst_kub(lst_x, lst_y, lst_spline)
        if print_all:
            print(*lst_k, sep='\n')
        print_kub(lst_x, lst_y, lst_spline)

        kub_sp(vvod_csv())
        
        
def __zadacha4_4__():
    '''Аппроксимация линейной функцией'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv

    def print_lin(lst_x, lst_y, stroka, dispers_lin):
        lst_x_new = np.linspace(np.min(lst_x), np.max(lst_x), 100)

        lst_y_new = [eval(stroka) for x in lst_x_new]

        plt.plot(lst_x, lst_y, 'o', lst_x_new, lst_y_new)
        plt.legend(['data', 'апрокс. фун.'], loc='best')
        plt.grid(True)
        plt.title(f"Апроксимирующая линейная функция:")
        plt.show()
        print(f"y = {stroka}")
        print(f'Величина дисперсии: {dispers_lin}')

    def lst_lin(lst_x, lst_y, stroka):
        lst_rez = []
        for i in range(len(lst_x)):
            x = lst_x[i]
            lst_rez.append([lst_x[i], lst_y[i], eval(stroka)])
        return lst_rez

    def MNK(lst, print_all=True):
        global dispers_lin
        global lst_tochki
        # Находим наше СЛАУ
        a11 = sum([el[0]**2 for el in lst])
        a12 = sum([el[0] for el in lst])
        a13 = sum([el[0] * el[1] for el in lst])
        a21 = sum([el[0] for el in lst])
        a22 = len(lst)
        a23 = sum([el[1] for el in lst])
        matrix = [[a11, a12, a13], [a21, a22, a23]]
        lst_c = reshit_gauss(matrix)
        c1 = lst_c[0]
        c0 = lst_c[1]
        stroka_rez = ''
        if c0 != 0:
            stroka_rez += f'{c1}*x'
            if c1 != 0:
                stroka_rez += f'+ {c0}'
        else:
            if c1 != 0:
                stroka_rez += f'{c0}'
        if print_all:
            print(f"Апроксимирующая линейная функция:  \n y = {stroka_rez}")

        lst_tochki = lst_lin([el[0] for el in lst], [el[1] for el in lst],\
                             stroka_rez)
        if print_all:
            print(*lst_tochki, sep='\n')

        dispers_lin = sum([(trio[1] - trio[2]) ** 2 for trio in lst_tochki])
    #     print(f'Величина дисперсии: {dispers}')

        if print_all:
            print_lin([el[0] for el in lst], [el[1] for el in lst],\
                      stroka_rez, dispers_lin)

        return(stroka_rez)

    MNK(vvod_csv())
    
    
def __zadacha4_5__():
    '''Аппроксимация квадратичной функцией'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv
    def print_kvad(lst_x, lst_y, stroka, dispers_kvad):
        lst_x_new = np.linspace(np.min(lst_x), np.max(lst_x), 100)

        lst_y_new = [eval(stroka) for x in lst_x_new]
    #     print(lst_for_rez)

        plt.plot(lst_x, lst_y, 'o', lst_x_new, lst_y_new)
        plt.legend(['data', 'апрокс. фун.'], loc='best')
        plt.grid(True)
        plt.title(f"Апроксимирующая квадратичная функция: ")
        plt.show()
        print(f"y = {stroka}")
        print(f'Величина дисперсии: {dispers_kvad}')

    def lst_kvad(lst_x, lst_y, stroka):
        lst_rez = []
        for i in range(len(lst_x)):
            x = lst_x[i]
            lst_rez.append([lst_x[i], lst_y[i], eval(stroka)])
        return lst_rez

    def aprox_kvad(lst, print_al=True):
        global dispers_kvad
        global lst_tochki_1
        # Находим наше СЛАУ
        a11 = sum([el[0] ** 4 for el in lst])
        a12 = sum([el[0] ** 3 for el in lst])
        a13 = sum([el[0] ** 2 for el in lst])
        a14 = sum([el[0] ** 2*el[1] for el in lst])

        a21 = sum([el[0] ** 3 for el in lst])
        a22 = sum([el[0] ** 2 for el in lst])
        a23 = sum([el[0] for el in lst])
        a24 = sum([el[0] * el[1] for el in lst])

        a31 = sum([el[0] ** 2 for el in lst])
        a32 = sum([el[0] for el in lst])
        a33 = len(lst)
        a34 = sum([el[1] for el in lst])

        matrix = [[a11, a12, a13, a14], [a21, a22, a23, a24],\
                  [a31, a32, a33, a34]]
        lst_c = reshit_gauss(matrix)
        c2 = lst_c[0]
        c1 = lst_c[1]
        c0 = lst_c[2]
        stroka_rez = ''
        if c2 != 0:
            stroka_rez += f'{c2}*x**2'
            if c1 != 0:
                stroka_rez += f'+ {c1}*x'
                if c0 != 0:
                    stroka_rez += f'+ {c0}'
            else:
                if c0 != 0:
                    stroka_rez += f'{c0}'
        else:
            if c1 != 0:
                stroka_rez += f'{c1}*x'
                if c0 != 0:
                    stroka_rez += f'+ {c0}'
            else:
                if c0 != 0:
                    stroka_rez += f'{c0}'
        if print_all:
            print(f"Апроксимирующая квадратичная функция: \n y = {stroka_rez}")

        lst_tochki_1 = lst_kvad([el[0] for el in lst], [el[1] for el in lst],\
                                stroka_rez)
        if print_all:
            print(*lst_tochki_1, sep='\n')
        dispers_kvad = sum([(trio[1]-trio[2]) ** 2 for trio in lst_tochki_1])
    #     print(f'Величина дисперсии: {dispers}')

        if print_all:
            print_kvad([el[0] for el in lst], [el[1] for el in lst],\
                       stroka_rez, dispers_kvad)
        print(lst_tochki_1)
        return(stroka_rez)

    aprox_kvad(vvod_csv())


def __zadacha4_6__():
    '''Аппроксимация ф-ей нормального распределения'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv
    def print_apr_norm(lst, stroka, dispers_norm):
        lst_x = [el[0] for el in lst]
        lst_y = [el[1] for el in lst]

        lst_x_new = np.linspace(np.min(lst_x), np.max(lst_x), 100)
        lst_y_new = [eval(stroka) for x in lst_x_new]

        plt.plot(lst_x, lst_y, 'o', lst_x_new, lst_y_new)
        plt.legend(['data', 'апрокс. фун.'], loc='best')
        plt.grid(True)
        plt.title(f"Аппроксимация функцией нормального распределения")
        plt.show()
        print(f"y = {stroka}")
        print(f'Величина дисперсии: {dispers_norm}')

    def apr_norm(lst, print_all=True):
        global lst_points
        global dispers_norm
        lst_x = [el[0] for el in lst]
        lst_y = [el[1] for el in lst]

        matrix_x = []
        matrix_y = []
        for ind, el in enumerate(lst_x):
            matrix_x.append([1, lst_x[ind], lst_x[ind] ** 2])
            if lst_y[ind] >= 0:
                matrix_y.append([log(lst_y[ind] + 0.1)])
            else:
                lst_y[ind] = fabs(lst_y[ind])
                matrix_y.append([log(lst_y[ind] + 0.1)])

        B = dot(dot(linalg.inv(dot(array(matrix_x).T, array(matrix_x))),\
                    array(matrix_x).T), array(matrix_y))

        a = exp(B[0][0] - (B[1][0] ** 2) / (4 * B[2][0]))
        b = - (1 / (B[2][0]))
        c = - (B[1][0] / (B[2][0] * 2))
    #     print(a, b, c)

        stroka = f'{a} * {math.e}**(-((x-{c})**2)/({b}))'

    #     stroka = str(f1.subs({a:aa, c:cc, b:bb}))
    #     print(stroka)

        lst_points = []
        for i, x in enumerate(lst_x):
            lst_points.append([lst_x[i], lst_y[i], eval(stroka)])
        if print_all:
            print(*lst_points, sep='\n')
        # Считаем дисперсию
        dispers_norm = sum([(trio[1] - trio[2]) ** 2 for trio in lst_points])

        if print_all:
            print_apr_norm(lst, stroka, dispers_norm)
        return stroka

    apr_norm(lst)

    def biggest(num1, num2, num3):
        if(num1 <= num2 and num1 <= num3):
            MNK()
        elif(num2 <= num1 and num2 <= num3): 
            aprox_kvad()
        else:
            apr_norm()
    biggest(dispers_lin, dispers_kvad, dispers_norm)
    apr_norm(vvod_csv())
    
def __zadacha5_1__():
    '''Расчет АХЧ'''
    def F(lst): 
        def vvod_csv():
            FILENAME = input('Введите название файла: ')
        
            lst_csv = []
            with open(FILENAME, newline='') as f:  
                reader = csv.reader(f)
                for row in reader:
                    lst_csv.append([float(el) for el in row])
                   
            return lst_csv
    
        period = (len(lst)-1)/(lst[-1][0] - lst[0][0])
        spektor = []
        for i in range(1, len(lst)//2):
            y = (rfft([el[1] for el in lst]))[i]
            x = rfftfreq(len(lst), 1/period)[i]
            spektor.append([x,y])
    
        #Считаем коэффициент несинусоидальности 
        peaks_n = (rfft([el[1] for el in lst]))
        peaks = list(peaks_n)
    #     peaks = np.abs(rfft)
        step = spektor[1][0] - spektor[0][0]
        ind = peaks.index(max(peaks))
        S0 = ((peaks[ind-1] + peaks[ind+1])/2 + peaks[ind])*step
    #     print(S0)
        Si = 0
        for i in range(1, len(peaks)):
            Si += (peaks[i-1] + peaks[i])*step/2
        Si = abs(Si - S0)
        print(f'Коэффициент несинусоидальности {Si/S0}')
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot([el[0] for el in lst], [el[1] for el in lst])
        axs[0].grid(True)
        axs[0].set_title(f"Исходная функция")
        axs[0].set_xlabel(f"x")
        axs[0].set_ylabel(f"y")
        
        axs[1].plot([el[0] for el in spektor], [el[1] for el in spektor])
        axs[1].grid(True)
        axs[1].set_title(f"Спектрограмма")
        axs[1].set_xlabel('Частота (Гц)')
        axs[1].set_ylabel('Точки (по y)')
        
        recover = irfft(peaks)
        axs[2].plot(recover)
        axs[2].grid(True)
        axs[2].set_title(f"Восстановленная функция")
        axs[2].set_xlabel(f"x")
        axs[2].set_ylabel(f"y")
        
        #Считаем дисперсию
        begin_y = [el[1] for el in lst]
        dispers = sum\
            ([(begin_y[i]-recover[i])**2 for i in range(len(recover))])
        print(f'Величина дисперсии: {dispers}')
        
    F(vvod_csv())
    
    
def __zadacha5_2__():
    '''Эксперимент с 10%'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv
    
    def F_experiment(lst): 
    
        period = (len(lst)-1)/(lst[-1][0] - lst[0][0])
        spektor = []
        for i in range(1, len(lst)//2):
            y = (rfft([el[1] for el in lst]))[i]
            x = rfftfreq(len(lst), 1/period)[i]
            spektor.append([x,y])
    
        #Считаем коэффициент несинусоидальности 
        peaks_n = (rfft([el[1] for el in lst]))
        peaks = list(peaks_n)
        step = spektor[1][0] - spektor[0][0]
        ind = peaks.index(max(peaks))
        S0 = ((peaks[ind-1] + peaks[ind+1])/2 + peaks[ind])*step
        Si = 0
        for i in range(1, len(peaks)):
            Si += (peaks[i-1] + peaks[i])*step/2
        Si = abs(Si - S0)
        print(f'Коэффициент несинусоидальности {Si/S0}')
        
        part = len(lst)//10
        part_sp = len(spektor)//10
        fig, axs = plt.subplots(10, 3, figsize=(15, 30))
        plt.subplots_adjust(wspace=2, hspace=0.7)
        def plott(i, n, n_sp):
    
            axs[i-1, 0].plot([el[0] for el in lst], [el[1] for el in lst])
            axs[i-1, 0].plot([el[0] for el in lst[:n]],\
                             [el[1] for el in lst[:n]], 'r')
            axs[i-1, 0].grid(True)
            axs[i-1, 0].set_title(f"Исходная функция (красным)")
            axs[i-1, 0].set_xlabel(f"x")
            axs[i-1, 0].set_ylabel(f"y")
    
            axs[i-1, 1].plot([el[0] for el in spektor[:n_sp]],\
                             [el[1] for el in spektor[:n_sp]])
            axs[i-1, 1].grid(True)
            axs[i-1, 1].set_title(f"Спектрограмма (отрезали с конца {(i-1)*10}%)")
            axs[i-1, 1].set_xlabel('Частота (Гц)')
            axs[i-1, 1].set_ylabel('Точки (по y)')
    
            recover = irfft(peaks[:n])
            axs[i-1, 2].plot(recover)
            axs[i-1, 2].grid(True)
            axs[i-1, 2].set_title(f"Восстановленная функция")
            axs[i-1, 2].set_xlabel(f"x")
            axs[i-1, 2].set_ylabel(f"y")
        
        for i in range(1,11):
            plott(i, part*(11-i), part_sp*(11-i))
        F_experiment(vvod_csv())
        
        
def __zadacha5_3__():
    '''Эксперимент с 10%'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv
    
    def F_experiment2(lst): 

        period = (len(lst)-1)/(lst[-1][0] - lst[0][0])
        spektor = []
        for i in range(1, len(lst)//2):
            y = (rfft([el[1] for el in lst]))[i]
            x = rfftfreq(len(lst), 1/period)[i]
            spektor.append([x,y])
    
        #Считаем коэффициент несинусоидальности 
        peaks_n = (rfft([el[1] for el in lst]))
        
        peaks = list(peaks_n)
        step = spektor[1][0] - spektor[0][0]
        ind = peaks.index(max(peaks))
        S0 = ((peaks[ind-1] + peaks[ind+1])/2 + peaks[ind])*step
    #     print(S0)
        Si = 0
        for i in range(1, len(peaks)):
            Si += (peaks[i-1] + peaks[i])*step/2
        Si = abs(Si - S0)
        print(f'Коэффициент несинусоидальности {Si/S0}')
        
        ten_percent = len(spektor)//10
        ten_percent_2 = len(peaks)//10
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot([el[0] for el in lst], [el[1] for el in lst])
        axs[0].grid(True)
        axs[0].set_title(f"Исходная функция")
        axs[0].set_xlabel(f"x")
        axs[0].set_ylabel(f"y")
        
        axs[1].plot([el[0] for el in spektor], [el[1] for el in spektor])
        axs[1].plot([el[0] for el in spektor[ten_percent:9*ten_percent]],\
                    [el[1] for el in spektor[ten_percent:9*ten_percent]], 'r')
        axs[1].grid(True)
        axs[1].set_title(f"Спектрограмма (по красному)")
        axs[1].set_xlabel('Частота (Гц)')
        axs[1].set_ylabel('Точки (по y)')
        
        recover = irfft(peaks[ten_percent_2:9*ten_percent_2])
        axs[2].plot(recover)
        axs[2].grid(True)
        axs[2].set_title(f"Восстановленная функция")
        axs[2].set_xlabel(f"x")
        axs[2].set_ylabel(f"y")
        
        #Считаем дисперсию
        begin_y = [el[1] for el in lst]
        dispers = sum\
            ([(begin_y[i]-recover[i])**2 for i in range(len(recover))])
        print(f'Величина дисперсии: {dispers}')
        F_experiment2(vvod_csv())
        
        
def __zadacha5_4__():
    '''Вейвлеты haar'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv
    def haar_wavelet(lst):
        # Строим график исходной функции
        plt.plot([el[0] for el in lst], [el[1] for el in lst], 'orange')
        plt.grid(True)
        plt.title(f"Исходная функция")
        plt.show()

        Y = [el[1] for el in lst]
        Y_w = pywt.wavedec(Y, 'haar', level=4)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].plot(Y_w[4])
        axs[0, 0].grid(True)
        axs[0, 0].set_title(f"Haar - 1 уровень")

        axs[0, 1].plot(Y_w[3])
        axs[0, 1].grid(True)
        axs[0, 1].set_title(f"Haar - 2 уровень")

        axs[1, 0].plot(Y_w[2])
        axs[1, 0].grid(True)
        axs[1, 0].set_title(f"Haar - 3 уровень")

        axs[1, 1].plot(Y_w[1])
        axs[1, 1].grid(True)
        axs[1, 1].set_title(f"Haar - 4 уровень")

        # Считаем коэффициент несинусоидальности
        d1 = sum([abs(el) for el in Y_w[1]])
        d2 = sum([abs(el) for el in Y_w[2]])
        d3 = sum([abs(el) for el in Y_w[3]])
        d4 = sum([abs(el) for el in Y_w[4]])

        da = sum([abs(i) for i in Y_w[0]])

        print(f"Коэффициент несинусоидальности {(d1 ** 2 + d2 ** 2 + d3 ** 2 + d4 ** 2) ** 0.5 / da}")
    haar_wavelet(vvod_csv()) 
    
    
def __zadacha5_5__():
    '''Вейвлеты db'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv

    def db_wavelet(lst):
        # Строим график исходной функции
        plt.plot([el[0] for el in lst], [el[1] for el in lst], 'orange')
        plt.grid(True)
        plt.title(f"Исходная функция")
        plt.show()

        Y = [el[1] for el in lst]
        Y_w = pywt.wavedec(Y, 'db1', level=4)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].plot(Y_w[4])
        axs[0, 0].grid(True)
        axs[0, 0].set_title(f"db - 1 уровень")

        axs[0, 1].plot(Y_w[3])
        axs[0, 1].grid(True)
        axs[0, 1].set_title(f"db - 2 уровень")

        axs[1, 0].plot(Y_w[2])
        axs[1, 0].grid(True)
        axs[1, 0].set_title(f"db - 3 уровень")

        axs[1, 1].plot(Y_w[1])
        axs[1, 1].grid(True)
        axs[1, 1].set_title(f"db - 4 уровень")

        # Считаем коэффициент несинусоидальности
        d1 = sum([abs(el) for el in Y_w[1]])
        d2 = sum([abs(el) for el in Y_w[2]])
        d3 = sum([abs(el) for el in Y_w[3]])
        d4 = sum([abs(el) for el in Y_w[4]])

        da = sum([abs(i) for i in Y_w[0]])

        print(f"Коэффициент несинусоидальности {(d1 ** 2 + d2 ** 2 + d3 **2 + d4 ** 2) ** 0.5 / da}")

    db_wavelet(vvod_csv())
    
    
def __zadacha5_6__():
    '''Мексиканская шляпа'''
    def vvod_csv():
        FILENAME = input('Введите название файла: ')

        lst_csv = []
        with open(FILENAME, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                lst_csv.append([float(el) for el in row])

        return lst_csv

    def w(a, b, t):
        # a- изменение временного масштаба
        # b- сдвиг во времени
        f = (1 / a ** 0.5) * math.exp(-0.5 * ((t - b) / a) ** 2) * (((t - b)\
                                                                     / a) **\
                                                                    2 - 1)
        return f

    def mex_wavelet(lst):
        plt.title("Вейвлет «Мексиканская шляпа»")
        x = [el[0] for el in lst]
        y = [w(1, 12, t) for t in x]
        plt.plot(x, y, label="$\psi(t)$ a=1,b=12")
        y = [w(2, 12, t) for t in x]
        plt.plot(x, y, label="$\psi_{ab}(t)$ a=2 b=12")
        y = [w(3, 12, t) for t in x]
        plt.plot(x, y, label="$\psi_{ab}(t)$ a=3 b=12")
        y = [w(4, 12, t) for t in x]
        plt.plot(x, y, label="$\psi_{ab}(t)$ a=4 b=12")
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    # Test_sin.csv
    mex_wavelet(vvod_csv())
    
    
def __zadacha5_7__():
    '''Вейвлеты'''
    y = np.append(0, np.array(pd.read_csv('Test_sin.csv'))[:, 1:])
    w = pywt.Wavelet('sym5')
    # найдем количество максимальных уровней декомпозиции
    num_levels = pywt.dwt_max_level(y.shape[0], filter_len=w.dec_len)
    df4 = np.append(0, np.array(pd.read_csv('2_sin_5_4.5.csv'))[:, 1:])

    def descrete_decomposition(num_levels, y, name_sample):
        for wev in ['haar', 'db2']:
            x = np.append(0, np.array(pd.read_csv(name_sample))[:, :1])
            plt.plot(x, y, 'orange')
            plt.grid(True)
            plt.title(f" {name_sample} : Исходная функция - {wev}")
            plt.show()
            for i in range(1, num_levels+1):
                # коэфициенты аппроксимации (выход фильтра нижних частот)
                coeffs = pywt.wavedec(y, wev, level=i)
                plt.subplot(2, 1, 1)
                plt.title(f'{name_sample} : Вейвлет - {wev}, декомпозиция\
                          {i}-ого уровня')
                plt.plot(coeffs[0], 'b', linewidth=2, label=f'cA, level={i}')
                plt.grid()
                plt.legend(loc='best')
                plt.show()
                # коэфициенты детализации (выход фильтра высоких частот)
                plt.subplot(2, 1, 2)
                plt.plot(coeffs[1], 'r', linewidth=2, label=f'cD, level={i}')
                plt.grid()
                plt.legend(loc='best')
                plt.show()
    descrete_decomposition(4, df4, '2_sin_5_4.5.csv')

    def count_level(func, wavelet):
        a, b = pywt.cwt(func, np.arange(1, 4), wavelet)
        return a, b

    def cont_decomposition(num_levels, y, name_sample):
        for wev in ['gaus1', 'mexh']:
            x = np.append(0, np.array(pd.read_csv(name_sample))[:, :1])
            plt.plot(x, y, 'orange')
            plt.grid(True)
            plt.title(f" {name_sample} : Исходная функция - {wev}")
            plt.show()
            for i in range(1, num_levels + 1):
                wavelet = pywt.ContinuousWavelet(wev, level=i)
                coef, freqs = count_level(y, wavelet)
                if i == 1:
                    plt.subplot(2, 1, 1)
                    plt.title(f' {name_sample}: Вейвлет - {wev}, декомпозиция\
                              {i}-ого уровня')
                    plt.plot(coef[0], label=f' level={i}')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.show()
                else:
                    for k in range(2, i + 1):
                        coef, freqs = count_level(coef[0], wavelet)
                    plt.subplot(2, 1, 1)
                    plt.title(f' {name_sample}: Вейвлет - {wev}, декомпозиция\
                              {i}-ого уровня')
                    plt.plot(coef[0], label=f' level={i}')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.show()
    cont_decomposition(4, df4, '2_sin_5_4.5.csv')

    def coeficcient(y, name_sample):
        for wev in ['haar', 'db2']:
            coeffs = pywt.wavedec(y, wev, level=4)
            d1 = sum([abs(el) for el in coeffs[1]])
            d2 = sum([abs(el) for el in coeffs[2]])
            d3 = sum([abs(el) for el in coeffs[3]])
            d4 = sum([abs(el) for el in coeffs[4]])
            da = sum([abs(i) for i in coeffs[0]])
            # дисперсия
            print(coeffs[0].shape)
            x = np.append(0, np.array(pd.read_csv(name_sample))[:, :1])
            variation = 0
            # for i in range(len(x)):
            # variation += (x[i]-coeffs[0][i])**2

            print(f" {name_sample} : Коэфф. несин. {wev} = {(d1 ** 2 + d2 **2 + d3 ** 2 + d4 ** 2) ** 0.5 / da}")
    coeficcient(df4, '2_sin_5_4.5.csv')
    wavelet = pywt.ContinuousWavelet('gaus1', level=4)

    def count_level(func, wavelet):
        a, b = pywt.cwt(func, np.arange(1, 5), wavelet)
        return a, b
    coef, freqs = count_level(df4, wavelet)
    coef[0]
    plt.plot(coef[0][10:-5])
    plt.show()
    plt.plot(coef[3][10:-5])


def __zadacha7__():
    '''Алгоритм имитации отжига'''

    sns.set(context="talk", style="darkgrid", palette="hls",\
            font="sans-serif", font_scale=1.05)
    FIGSIZE = (19, 8)  #: Figure size, in inches!
    mpl.rcParams['figure.figsize'] = FIGSIZE


    def annealing(random_start,
                  cost_function,
                  random_neighbour,
                  acceptance,
                  temperature,
                  maxsteps=1000,
                  debug=True):
        state = random_start()
        cost = cost_function(state)
        states, costs = [state], [cost]
        for step in range(maxsteps):
            fraction = step / float(maxsteps)
            T = temperature(fraction)
            new_state = random_neighbour(state, fraction)
            new_cost = cost_function(new_state)
            if debug:
                print("Шаг #{:>2}/{:>2} : T = {:>4.3g}, Город = {:>4.3g},\
                      Стоимость = {:>4.3g}, Новый город = {:>4.3g},\
                          Новая стоимость = {:>4.3g} ...".format(step,\
                              maxsteps, T, state, cost, new_state, new_cost))
            if acceptance_probability(cost, new_cost, T) > rn.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
                # print("  ==> Accept it!")
            # else:
            #    print("  ==> Reject it...")
        return state, cost_function(state), states, costs

    interval = (-10, 10)


    def f(x):
        return x ** 2


    def clip(x):
        a, b = interval
        return max(min(x, b), a)


    def random_start():
        a, b = interval
        return a + (b - a) * rn.random_sample()


    def cost_function(x):
        return f(x)


    def random_neighbour(x, fraction=1):
        amplitude = (max(interval) - min(interval)) * fraction / 10
        delta = (-amplitude/2.) + amplitude * rn.random_sample()
        return clip(x + delta)


    def acceptance_probability(cost, new_cost, temperature):
        if new_cost < cost:
            return 1
        else:
            p = np.exp(- (new_cost - cost) / temperature)
            return p


    def temperature(fraction):
        return max(0.01, min(1, 1 - fraction))

    maxsteps = int(input('Введите количество городов: '))

    annealing(random_start, cost_function, random_neighbour,\
              acceptance_probability, temperature, maxsteps, debug=True)
        
        
        
def __zadacha7_3__():
    '''Муравьиная колония'''
    class ACA_TSP:
        def __init__(self, func, n_dim,
                     size_pop=10, max_iter=20,
                     distance_matrix=None,
                     alpha=1, beta=2, rho=0.1,
                     ):
            self.func = func
            self.n_dim = n_dim
            self.size_pop = size_pop
            self.max_iter = max_iter
            self.alpha = alpha
            self.beta = beta
            self.rho = rho

            self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 *\
                                             np.eye(n_dim, n_dim))

            self.Tau = np.ones((n_dim, n_dim))
            self.Table = np.zeros((size_pop, n_dim)).astype(np.int)
            self.y = None
            self.generation_best_X, self.generation_best_Y = [], []
            self.x_best_history, self.y_best_history =\
                self.generation_best_X, self.generation_best_Y
            self.best_x, self.best_y = None, None

        def run(self, max_iter=None):
            self.max_iter = max_iter or self.max_iter
            for i in range(self.max_iter):
                prob_matrix = (self.Tau ** self.alpha) *\
                    (self.prob_matrix_distance) ** self.beta
                for j in range(self.size_pop):
                    self.Table[j, 0] = 0  # Начальная точка
                    for k in range(self.n_dim - 1):
                        taboo_set = set(self.Table[j, :k + 1])
                        allow_list = list(set(range(self.n_dim)) - taboo_set)
                        prob = prob_matrix[self.Table[j, k], allow_list]
                        prob = prob / prob.sum()
                        next_point = np.random.choice(allow_list,\
                                                      size=1, p=prob)[0]
                        self.Table[j, k + 1] = next_point

                # Рассчет расстояния
                y = np.array([self.func(i) for i in self.Table])

                # Сохраняем лучший результат
                index_best = y.argmin()
                x_best, y_best = self.Table[index_best, :].copy(),\
                    y[index_best].copy()
                self.generation_best_X.append(x_best)
                self.generation_best_Y.append(y_best)

                # Рассчитываем количество феромона, которое нужно\
                #нанести заново
                delta_tau = np.zeros((self.n_dim, self.n_dim))
                for j in range(self.size_pop):
                    for k in range(self.n_dim - 1):
                        n1, n2 = self.Table[j, k], self.Table[j, k + 1]
                        delta_tau[n1, n2] += 1 / y[j]
                    n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]
                    delta_tau[n1, n2] += 1 / y[j]

                # Феромоны
                self.Tau = (1 - self.rho) * self.Tau + delta_tau

            best_generation = np.array(self.generation_best_Y).argmin()
            self.best_x = self.generation_best_X[best_generation]
            self.best_y = self.generation_best_Y[best_generation]
            return self.best_x, self.best_y

        fit = run
    num_points = int(input('Введите количество городов: '))
    points_coordinate = np.random.rand(num_points, 2) 
    distance_matrix = spatial.distance.cdist(points_coordinate,\
                                             points_coordinate, metric='euclidean')

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points],\
                                    routine[(i + 1) % num_points]] for i\
                    in range(num_points)])

    def main():
        aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                      size_pop=3, max_iter=10,
                      distance_matrix=distance_matrix)
        best_x, best_y = aca.run()
        # Plot the result
        fig, ax = plt.subplots(1, 2)
        best_points_ = np.concatenate([best_x, [best_x[0]]])
        best_points_coordinate = points_coordinate[best_points_, :]
        for index in range(0, len(best_points_)):
            ax[0].annotate(best_points_[index], \
                           (best_points_coordinate[index, 0], \
                            best_points_coordinate[index, 1]))
        ax[0].plot(best_points_coordinate[:, 0],\
                   best_points_coordinate[:, 1], 'o-r')
        pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
        plt.show()
        print('Кратчайший путь:', best_x, 'Стоимость', best_y)

    if __name__ == "__main__":
        main()
        
        
def diff_left_side(x, y):
    h = x[1] - x[0]
    dy = []
    for i in range(1, len(x)):
        dy.append((y[i] - y[i - 1]) / h)
    return dy


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        try:
            complex(str)
            return True
        except ValueError:
            return False
    except TypeError:
        try:
            complex(str)
            return True
        except TypeError:
            return False
        

def type_conversion(str):
    try:
        float(str)
        return float(str)
    except ValueError:
        complex(str)
        return complex(str)
    
    
def matrixTranspose(anArray):
    transposed = [None] * len(anArray[0])
    for t in range(len(anArray)):
        transposed[t] = [None] * len(anArray)
        for tt in range(len(anArray[t])):
            transposed[t][tt] = anArray[tt][t]
    return transposed


def matrix(random=0, float_random=0, a=1, b=100):
    m = input('Введите количество строк: ')

    while m.isdigit() != 1:
        print("Неверный формат ввода")
        m = input('Введите количество строк: ')
    n = input('Введите количество столбцов: ')

    while n.isdigit() != 1:
        print("Неверный формат ввода")
        n = input('Введите количество столбцов: ')
    m = int(m)
    n = int(n)
    matr = []
    if random == 0:
        for i in range(m):
            t = []
            for j in range(n):
                _ = input(f'Введите элемент {i + 1} \
                          строки {j + 1} столбца: ')
                while is_number(_) != 1:
                    print("Неверный формат ввода")
                    _ = input(f'Введите элемент {i + 1} \
                              строки {j + 1} столбца: ')
                try:
                    t.append(float(_))
                except ValueError:
                    try:
                        t.append(complex(_))
                    except ValueError:
                        None
            matr.append(t)
    else:
        for i in range(m):
            t = []
            for j in range(n):
                if float_random == 1:
                    _ = uniform(a, b)
                    t.append(_)
                else:
                    _ = randint(a, b)
                    t.append(_)
            matr.append(t)

    return matr


def det2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]


def alg_dop(matrix, somme=None, prod=1):
    if (somme == None):
        somme = []
    if (len(matrix) == 1):
        somme.append(matrix[0][0])
    elif (len(matrix) == 2):
        somme.append(det2(matrix) * prod)
    else:
        for index, elmt in enumerate(matrix[0]):
            transposee = [list(a) for a in zip(*matrix[1:])]
            del transposee[index]
            mineur = [list(a) for a in zip(*transposee)]
            somme = alg_dop(mineur, somme, prod * matrix[0][index] * (-1) ** (index + 2))
    return somme


def determinant(matrix):
    return sum(alg_dop(matrix))


def sum_matrix(mtrx_1, mtrx_2):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            t = type_conversion(mtrx_1[i][j])
            m = type_conversion(mtrx_2[i][j])
            tmp_mtrx[i][j] = t + m
    return tmp_mtrx


def minor(matrix, i, j):
    minor = []
    for q in (matrix[:i] + matrix[i + 1:]):
        _ = q[:j] + q[j + 1:]
        minor.append(_)
    return minor


def subtraction_matrix(mtrx_1, mtrx_2):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            t = type_conversion(mtrx_1[i][j])
            m = type_conversion(mtrx_2[i][j])
            tmp_mtrx[i][j] = t - m
    return tmp_mtrx


def mult_by_count_matrix(mtrx_1, k):
    tmp_mtrx = [[0 for j in range(len(mtrx_1))] for i in range(len(mtrx_1[0]))]
    for i in range(len(mtrx_1)):
        for j in range(len(mtrx_1[0])):
            k = type_conversion(k)
            t = type_conversion(mtrx_1[i][j])
            tmp_mtrx[i][j] = t * k
    return tmp_mtrx


def multiply_matrix(mtrx_1, mtrx_2):
    s = 0
    t = []
    m3 = []
    r1 = len(mtrx_1)
    for z in range(0, r1):
        for j in range(0, r1):
            for i in range(0, r1):
                l1 = type_conversion(mtrx_1[z][i])
                l2 = type_conversion(mtrx_2[i][j])
                s = s + l1 * l2
            t.append(s)
            s = 0
        m3.append(t)
        t = []
    return m3
    return tmp_mtrx


def single_variable(row,
                    index):
    return ([(-i / row[index]) for i in (row[:index] +\
                                         row[index + 1:-1])] + \
            [row[-1] / row[index]])


def norma(matrix):
    norma_matrix = []
    for i in range(len(matrix)):
        summa = 0
        for j in range(len(matrix)):
            summa += abs(matrix[i][j])
        norma_matrix.append(summa)
    return max(norma_matrix)


def reverse_matrix(matrix):
    deter = determinant(matrix)
    try:
        a = 1 / deter
    except ZeroDivisionError:
        return 'Нулевой определитель'
    matr_dop = [[0] * len(matrix) for i in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matr_dop[i][j] = (-1) ** (i + j) * determinant(minor(matrix, i, j))
    matr_dop_T = matrixTranspose(matr_dop)
    return mult_by_count_matrix(matr_dop_T, a)


def cord(matrix):
    return (norma(matrix) * norma(reverse_matrix(matrix)))


def method_Jacobi(a, b):
    eps = float(input('Погрешность: '))
    matrix = []
    for j in range(len(b)):
        matrix.append(a[j] + b[j])
    interm = [0] * (len(matrix)) + [1]
    variables = [0] * len(matrix)
    k = -1
    interm_2 = [0] * (len(matrix)) + [1]
    count = 0
    while k != 0:
        k = 0
        for i in range(len(matrix)):
            variables[i] = single_variable(matrix[i], i)
            for j in range(len(matrix)):
                ne_know = (interm[:i] + interm[i + 1:])
                interm_2[i] += variables[i][j] * ne_know[j]
            if abs(interm[i] - interm_2[i]) > eps:
                k += 1
        interm = interm_2
        interm_2 = [0] * (len(matrix)) + [1]
        # print(interm[:-1])
        # print(k)
        # print('____')
        count += 1
        if count == 1000:
            return (['Метод Якоби не смог найти решение!'])
    return (a, reverse_matrix(a), interm[:-1])


def pick_nonzero_row(m, k):
    while k < m.shape[0] and not m[k, k]:
        k += 1
    return k

def gssjrdn(a, b):
    nc = []
    for i in range(len(a)):
        nc.append(a[i])
    a = np.array(a, float)
    b = np.array(b, float)
    n = len(b)
    st = a

    m = np.hstack((st,
                   np.matrix(np.diag([1.0 for i in range(st.shape[0])]))))
    for k in range(n):
        swap_row = pick_nonzero_row(m, k)
        if swap_row != k:
            m[k, :], m[swap_row, :] = m[swap_row, :], np.copy(m[k, :])
        if m[k, k] != 1:
            m[k, :] *= 1 / m[k, k]
        for row in range(k + 1, n):
            m[row, :] -= m[k, :] * m[row, k]
    for k in range(n - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if m[row, k]:
                m[row, :] -= m[k, :] * m[row, k]

    for k in range(n):
        if np.fabs(a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    for j in range(k, n):
                        a[k, j], a[i, j] = a[i, j], a[k, j]
                    b[k], b[i] = b[i], b[k]
                    break
        pivot = a[k, k]
        for j in range(k, n):
            a[k, j] /= pivot
        b[k] /= pivot
        for i in range(n):
            if i == k or a[i, k] == 0:
                continue
            factor = a[i, k]
            for j in range(k, n):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]

    return nc, np.hsplit(m, n // 2)[0], b


def frkgssjrdn(a, b):
    nc = []
    for i in range(len(a)):
        nc.append(a[i])
    a = np.array(a, float)
    b = np.array(b, float)
    n = len(b)

    for i in range(n):
        for j in range(n):
            a[i, j] = Fraction(a[i, j])
            b[i] = Fraction(*b[i])

    matrix = []
    for j in range(n):
        matrix.append(a[j] + b[j])
    matrix = np.array(matrix, float)
    matrix[i, j] = Fraction(matrix[i, j])

    for k in range(n):
        if np.fabs(a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    for j in range(k, n):
                        a[k, j], a[i, j] = a[i, j], a[k, j]
                    b[k], b[i] = b[i], b[k]
                    break
        pivot = a[k, k]
        for j in range(k, n):
            a[k, j] /= pivot
        b[k] /= pivot
        for i in range(n):
            if i == k or a[i, k] == 0:
                continue
            factor = a[i, k]
            for j in range(k, n):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]

    m = np.hstack((matrix,
                   np.matrix(np.diag([1.0 for i in range(matrix.shape[0])]))))
    for k in range(n):
        swap_row = pick_nonzero_row(m, k)
        if swap_row != k:
            m[k, :], m[swap_row, :] = m[swap_row, :], np.copy(m[k, :])
        if m[k, k] != 1:
            m[k, :] *= 1 / m[k, k]
        for row in range(k + 1, n):
            m[row, :] -= m[k, :] * m[row, k]
    for k in range(n - 1, 0, -1):
        for row in range(k - 1, -1, -1):
            if m[row, k]:
                m[row, :] -= m[k, :] * m[row, k]
    return nc, np.hsplit(m, n // 2)[0], b


def method_lin(x, y):
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x2_y = 0
    x_y = 0
    y_1 = 0
    for i in range(len(x)):
        x_1 += x[i]
        x_2 += x[i] ** 2
        x_3 += x[i] ** 3
        x_4 += x[i] ** 4
        x2_y += y[i] * x[i] ** 2
        x_y += y[i] * x[i]
        y_1 += y[i]
    n = len(x)
    a = [[x_2, x_3, x_4], [x_1, x_2, x_3], [n, x_1, x_2]]
    b = [[x2_y], [x_y], [y_1]]
    roots = gssjrdn(a, b)[2]
    c = []
    for i in range(2):
        c.append(*roots[i])

    def f_x(t):
        return c[1] * t + c[0]

    gamma = 0
    f = []
    for i in range(len(x)):
        f.append(f_x(x[i]))
        gamma += (y[i] - f[i]) ** 2
    exp = ''
    for i in [1, 0]:
        if c[i] != 0:
            exp += f'{c[i]}*t**{i} + '
    exp = exp[:-2]
    output = [[x[i]] + [y[i]] + [f[i]] for i in range(len(x))]
    return (output, exp, gamma)


def method_min_square(x, y):
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x2_y = 0
    x_y = 0
    y_1 = 0
    for i in range(len(x)):
        x_1 += x[i]
        x_2 += x[i] ** 2
        x_3 += x[i] ** 3
        x_4 += x[i] ** 4
        x2_y += y[i] * x[i] ** 2
        x_y += y[i] * x[i]
        y_1 += y[i]
    n = len(x)
    a = [
        [x_2, x_3, x_4],
        [x_1, x_2, x_3],
        [n, x_1, x_2]
    ]
    b = [
        [x2_y],
        [x_y],
        [y_1]
    ]
    roots = gssjrdn(a, b)[2]
    c = []
    for i in range(3):
        c.append(*roots[i])

    def f_x(t):
        return c[2] * t ** 2 + c[1] * t + c[0]

    gamma = 0
    f = []
    for i in range(len(x)):
        f.append(f_x(x[i]))
        gamma += (y[i] - f[i]) ** 2
    exp = ''
    for i in [2, 1, 0]:
        if c[i] != 0:
            exp += f'{c[i]}*t**{i} + '
    exp = exp[:-2]
    output = [[x[i]] + [y[i]] + [f[i]] for i in range(len(x))]
    return (output, exp, gamma)


def lagranz(x, y):
    t = symbols('t')
    z = 0
    for j in range(len(y)):
        numerator = 1;
        denominator = 1;
        for i in range(len(x)):
            if i == j:
                numerator = numerator * 1;
                denominator = denominator * 1
            else:
                numerator = expand(numerator * (t - x[i]))
                denominator = denominator * (x[j] - x[i])
        z = expand(z + y[j] * numerator / denominator)
    f_x = lambdify(t, z)
    output = []
    for k in range(len(x)):
        output.append([x[k], y[k], f_x(x[k])])
    return (output, z)


class SplineTuple:
    def __init__(self, a, b, c, d, x):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x
        
        
def BuildSpline(x, y):
    # Инициализация массива сплайнов
    n = len(x)
    splines = [SplineTuple(0, 0, 0, 0, 0) for _ in range(0, n)]
    for i in range(0, n):
        splines[i].x = x[i]
        splines[i].a = y[i]

    splines[0].c = splines[n - 1].c = 0.0

    # Вычисление прогоночных коэффициентов - прямой ход метода прогонки
    alpha = [0.0 for _ in range(0, n - 1)]
    beta = [0.0 for _ in range(0, n - 1)]

    for i in range(1, n - 1):
        hi = x[i] - x[i - 1]
        hi1 = x[i + 1] - x[i]
        A = hi
        C = 2.0 * (hi + hi1)
        B = hi1
        F = 6.0 * ((y[i + 1] - y[i]) / hi1 - (y[i] - y[i - 1]) / hi)
        z = (A * alpha[i - 1] + C)
        alpha[i] = -B / z
        beta[i] = (F - A * beta[i - 1]) / z

    # Нахождение решения - обратный ход метода прогонки
    for i in range(n - 2, 0, -1):
        splines[i].c = alpha[i] * splines[i + 1].c + beta[i]

    # По известным коэффициентам c[i] находим значения b[i] и d[i]
    for i in range(n - 1, 0, -1):
        hi = x[i] - x[i - 1]
        splines[i].d = (splines[i].c - splines[i - 1].c) / hi
        splines[i].b = hi * (2.0 * splines[i].c + splines[i - 1].c) /\
            6.0 + (y[i] - y[i - 1]) / hi
    return splines


def Interpolate(splines, x):
    if not splines:
        return None  # Если сплайны ещё не построены - возвращаем NaN

    n = len(splines)
    s = SplineTuple(0, 0, 0, 0, 0)

    if x <= splines[0].x:  # Если x меньше точки сетки x[0] -
    #пользуемся первым эл-тов массива
        s = splines[0]
    elif x >= splines[n - 1].x:  # Если x больше точки сетки x[n - 1] -
    #пользуемся последним эл-том массива
        s = splines[n - 1]
    else:  # Иначе x лежит между граничными точками сетки -
    #производим бинарный поиск нужного эл-та массива
        i = 0
        j = n - 1
        while i + 1 < j:
            k = i + (j - i) // 2
            if x <= splines[k].x:
                j = k
            else:
                i = k
        s = splines[j]

    dx = x - s.x
    return s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx;


accuracy = 0.00001
START_X = -1
END_X = 6
START_Y = -1
END_Y = 20
temp = []


def whence_differences(y_array):
    return_array = []
    for i in range(0, len(y_array) - 1):
        return_array.append(y_array[i + 1] - y_array[i])
    return return_array

def witchcraft_start(y_array, h):
    part_y = [y_array[0]]
    y = y_array
    for i in range(0, len(y_array) - 1):
        y = whence_differences(y)
        part_y.append(y[0] / math.factorial(i + 1) / (h ** (i + 1)))
    return part_y

def tragic_magic(coefficients_y, point, x_array):
    value = coefficients_y[0]
    for i in range(1, len(coefficients_y)):
        q = 1
        for j in range(0, i):
            q *= (point - x_array[j])
        value += coefficients_y[i] * q
    return value

def build_points(x_array, y_array):
    for i in range(0, len(x_array)):
        plt.scatter(x_array[i], y_array[i])

def newton_there(x_array, y_array):
    x0 = x_array[0]
    h = x_array[1] - x_array[0]
    build_points(x_array, y_array)
    part_y = witchcraft_start(y_array, h)
    x = np.linspace(x_array[0], x_array[len(x_array) - 1], 228)
    return (x, tragic_magic(part_y, x, x_array), part_y)

def witchcraft_continue(y_array, h):
    part_y = [y_array[len(y_array) - 1]]
    y = y_array
    for i in range(0, len(y_array) - 1):
        y = whence_differences(y)
        part_y.append(y[len(y) - 1] / math.factorial(i + 1) / (h ** (i + 1)))
    return part_y

def ecstatic_magic(coefficients_y, point, x_array):
    value = coefficients_y[0]
    for i in range(1, len(coefficients_y)):
        q = 1
        for j in range(0, i):
            q *= (point - x_array[len(x_array) - j - 1])
        value += coefficients_y[i] * q
    return value

def newton_here_the_boss(x_array, y_array):
    x0 = x_array[0]
    h = x_array[1] - x_array[0]
    build_points(x_array, y_array)
    part_y = witchcraft_continue(y_array, h)
    x = np.linspace(x_array[0], x_array[len(x_array) - 1], 228)
    return (x, ecstatic_magic(part_y, x, x_array), part_y)

def approximate_log_function(x, y):

    C = np.arange(0.01, 1, step = 0.01)
    a = np.arange(0.01, 1, step = 0.01)
    b = np.arange(0.01, 1, step = 0.01)

    min_mse = 9999999999
    parameters = [0, 0, 0]

    for i in np.array(np.meshgrid(C, a, b)).T.reshape(-1, 3):
        y_estimation = i[0] * np.log(i[1] * np.array(x) + i[2])
        mse = mean_squared_error(y, y_estimation)
        if mse < min_mse:
            min_mse = mse
            parameters = [i[0], i[1], i[2]]
    output = [[x[i]]+[y[i]]+[y_estimation[i]] for i in range(len(x))]
    return (output, min_mse)


def ap_norm_rasp(x,y_real):
    def norm(x, mean, sd):
        norm = []
        for i in range(len(x)):
            norm += [1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x[i] -\
                                                        mean)**2/(2*sd**2))]
        return np.array(norm)
    mean1, mean2 = 10, -2
    std1, std2 = 0.5, 10 
    m, dm, sd1, sd2 = [3, 10, 1, 1]
    p = [m, dm, sd1, sd2] # Initial guesses for leastsq
    y_init = norm(x, m, sd1) + norm(x, m + dm, sd2) # For final comparison plot
    def res(p, y, x):
        m, dm, sd1, sd2 = p
        m1 = m
        m2 = m1 + dm
        y_fit = norm(x, m1, sd1) + norm(x, m2, sd2)
        err = y - y_fit
        return err
    plsq = leastsq(res, p, args = (y_real, x))
    y_est = norm(x, plsq[0][0], plsq[0][2]) + norm(x, plsq[0][0] +\
                                                   plsq[0][1], plsq[0][3])
    gamma = 0
    for i in range(len(x)):
        gamma += (y_real[i]-y_init[i])**2
        output = [[x[i]]+[y_real[i]]+[y_init[i]] for i in range(len(x))]
    return (output, gamma)

def approximate_exp_function(x,y):
    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    x2_y = 0
    x_y = 0
    y_1 = 0
    for i in range(len(x)):
        x_1 += x[i]
        x_2 += x[i]**2
        x_3 += x[i]**3
        x_4 += x[i]**4
        x2_y += y[i]*x[i]**2
        x_y += y[i]*x[i]
        y_1 += y[i]
    n = len(x)
    a = [[x_2,x_3,x_4],[x_1,x_2,x_3],[n,x_1,x_2]]
    b = [[x2_y],[x_y],[y_1]]
    roots = gssjrdn(a,b)[2]
    c = []
    for i in range(3):
        c.append(*roots[i])
    def f_x(t):
        return c[0]*math.e**(c[1]*t)
    gamma = 0
    f = []
    for i in range(len(x)):
        f.append(f_x(x[i]))
        gamma += (y[i]-f[i])**2
    exp = ''
    for i in [1,0]:
        if c[i] != 0:
            exp += f'{c[i]}*e**({c[i+1]}*t) + '
    exp = exp[:-2]
    output = [[x[i]]+[y[i]]+[f[i]] for i in range(len(x))]
    return (output,exp, gamma)

def interpol_st(xp,fp):
    x = np.linspace(-np.pi, 10, 100)
    y = np.interp(x, xp, fp)
    fig, ax = plt.subplots()
    plt.plot(xp, fp,'b',label="Исходные точки")
    plt.plot(x, y,'r',label=f'Интерполированная функция')
    plt.title('Интерполяция точек стандартной numpy')
    plt.show()
    
def aprox_st(xp,fp):
    x = np.linspace(-len(xp)/2, len(xp)/2, 100)
    y = np.polyfit(xp, fp, 99)
    fig, ax = plt.subplots()
    plt.plot(xp, fp,'b',label="Исходные точки")
    plt.plot(x, y,'r',label=f'Интерполированная функция')
    plt.title('Аппроксимация точек стандартной numpy')
    plt.show()
    
    
def input_function():
    t = symbols('t')
    y = symbols('y')
    z = symbols('z')
    equation = []

    qwe = eval(input('Введите y` [доступные символы t,y,z]: '))
    equation.append(qwe)
    global answer
    answer = input('Хотите ввести еще одно уравнение? [1/0] ')
    if answer == '1':
        qwe = eval(input('Введите z` [доступные символы t,y,z]: '))
        equation.append(qwe)
    function = []
    for i in range(len(equation)):
        function.append(lambdify([t, y, z], equation[i]))
    return function


def euler(func, n=100):
    Y0 = float(input('Начальное условие, y0 = '))
    a = float(input('Начало промежутка x: '))
    b = float(input('Конец промежутка x: '))
    if answer == '1':
        Z0 = float(input('Начальное условие, z0 = '))
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    y[0] = Y0
    if answer == '1':
        z[0] = Z0
    x[0] = a
    output = [[0, Y0]]
    dx = b / float(n)
    if answer == '0':
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](x[k], y[k],0)
            output.append([x[k], y[k]])
        return output
    if answer == '1':
        output = [[0,  Y0, Z0]]
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](x[k], z[k],0)
            z[k + 1] = z[k] + dx * func[1](x[k], z[k],0)
            output.append([x[k], y[k],z[k]])
        return output
    

def euler_Koshi(func,n=100):
    """Решение ОДУ u'=f(y,x), начальное условие y(0) = U0 , c n шагами, \
        пока  x = b - конец отрезка интегрирования."""
    Y0 = float(input('Начальное условие, y0 = '))
    a = float(input('Начало промежутка x: '))
    b = float(input('Конец промежутка x: '))
    if answer == '1':
        Z0 = float(input('Начальное условие, z0 = '))
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)
    y[0] = Y0
    if answer == '1':
        z[0] = Z0
    x[0] = a
    output = [[0, Y0]]
    dx = b / n
    if answer == '0':
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](x[k], y[k],0)
            y[k + 1] = y[k] + dx/2*( func[0](x[k], y[k],0)+  func[0](x[k+1],\
                                                                     y[k+1],0))
            output.append([x[k], y[k]])
        return output
    if answer == '1':
        output = [[0,  Y0, Z0]]
        for k in range(n):
            x[k + 1] = x[k] + dx
            y[k + 1] = y[k] + dx * func[0](x[k], y[k],z[k])
            y[k + 1] = y[k] + dx/2*( func[0](x[k], y[k],0)+  func[0](x[k+1],\
                                                                     y[k+1],0))
            z[k + 1] = z[k] + dx * func[1](x[k],z[k],0)
            z[k + 1] = z[k] + dx/2*( func[1](x[k],z[k],0)+  func[1](x[k+1],\
                                                                    z[k+1],0))
            output.append([x[k], y[k],z[k]])
        return output
    
    
def rungekutta4(function):
    y0 = float(input('Начальное условие, y0 = '))    
    a = float(input('Начало промежутка x: '))
    b = float(input('Конец промежутка x: '))
    if answer == '1':
        z0 = float(input('Начальное условие, z0 = '))
        zn = z0
    n = 100
    h = (b - a) / n
    output = [[0, 0, y0]]
    counter = 0
    yn = y0
    otrezok = []
    for i in range(n):
        a += h
        otrezok.append(round(a, 5))
    if answer == '0':
        for i in otrezok:
            counter += 1
            k1 = function[0](i, yn,1) * h
            k2 = function[0](i + h / 2, yn + k1 / 2, 1) * h
            k3 = function[0](i + h / 2, yn + k2 / 2, 1) * h
            k4 = function[0](i + h, yn + k3, 1) * h
            yn = yn + (1 / 6) * (k1 + 2 * k2 + 3 * k3 + k4)
            output.append([counter, i, yn])
        return output
    elif answer == '1':
        output = [[0, 0, y0, z0]]
        for i in otrezok:
            counter += 1
            k1 = function[0](i, yn, zn) * h
            m1 = function[1](i, yn, zn) * h

            k2 = function[0](i + h / 2, yn + k1 / 2, zn + m1 / 2) * h
            m2 = function[1](i + h / 2, yn + k1 / 2, zn + m1 / 2) * h

            k3 = function[0](i + h / 2, yn + k2 / 2, zn + m2 / 2) * h
            m3 = function[1](i + h / 2, yn + k2 / 2, zn + m2 / 2) * h

            k4 = function[0](i + h, yn + k3, zn + m3) * h
            m4 = function[1](i + h, yn + k3, zn + m3) * h

            yn = yn + (1 / 6) * (k1 + 2 * k2 + 3 * k3 + k4)
            zn = zn + (1 / 6) * (m1 + 2 * m2 + 3 * m3 + m4)
            output.append([counter, i, yn, zn])
        return output
    

def __zadacha6_0__():    
    '''ОДУ'''
    plt.style.use('ggplot')    
    
    Input =  input_function()
    qwe = euler(Input)
    X1= []
    Y1 = []
    Z1 = []
    
    for i in range(len(qwe)):
        X1.append(qwe[i][0])
        Y1.append(qwe[i][1])
        if answer == '1':
            Z1.append(qwe[i][2])
    plt.plot(X1, Y1, label='Эйлер')
    if answer == '1':
        plt.plot(X1, Z1, label='Эйлер')
    plt.title("Эйлер")
    plt.legend(loc='best', prop={'size': 8}, frameon = False)
    plt.show()
    
    noname1 = method_min_square(X1,Y1)
    if answer == '1':
        noname2 = method_min_square(X1,Z1)
    _1 = noname1[0]
    if answer == '1':
        _2 = noname2[0]
    gamma1 = round(noname1[2],3)
    if answer == '1':
        gamma2 = round(noname2[2],3)
    x1_square = []
    f1_square = []
    x2_square = []
    f2_square = []
    for i in range(len(X1)):
        x1_square.append(_1[i][0])
        f1_square.append(_1[i][2])
        if answer == '1':
            x2_square.append(_2[i][0])
            f2_square.append(_2[i][2])
    plt.plot(x1_square, f1_square, 'r', label=f'Интерполированная функция\
             c G = {gamma1}')
    if answer == '1':
        plt.plot(x2_square, f2_square, 'r', label=f'Интерполированная \
                 функция c G = {gamma2}')
    plt.title("МНК Эйлера ")
    plt.legend(loc='best', prop={'size': 8}, frameon = False)
    plt.show()
    
    noname1 = newton_here_the_boss(X1,Y1)
    if answer == '1':
        noname2 = newton_here_the_boss(X1,Z1)
    x1_square = []
    f1_square = []
    x2_square = []
    f2_square = []
    for i in range(len(noname1[1])):
        x1_square.append(noname1[0][i])
        f1_square.append(noname1[1][i])
        if answer == '1':
            x2_square.append(noname2[0][i])
            f2_square.append(noname2[1][i])
    plt.plot(x1_square, f1_square, 'r', label=f'Аппроксимированная функция')
    if answer == '1':
        plt.plot(x2_square, f2_square, 'r', label=f'Аппроксимированная \
                 функция')
    plt.title("Аппроксимация Ньтоном Эйлера ")
    plt.legend(loc='best', prop={'size': 8}, frameon = False)
    plt.show()
    
    qwe2 = euler_Koshi(Input)
    X2= []
    Y2 = []
    Z2 = []
    
    for i in range(len(qwe)):
        X2.append(qwe[i][0])
        Y2.append(qwe[i][1])
        if answer == '1':
            Z2.append(qwe[i][2])
    plt.plot(X2, Y2, label='Эйлер-Коши')
    if answer == '1':
        plt.plot(X2, Z2, label='Эйлер-Коши')
    plt.title("Эйлер-Коши")
    plt.legend(loc='best', prop={'size': 8}, frameon = False)
    plt.show()
    
    noname1 = method_min_square(X2,Y2)
    if answer == '1':
        noname2 = method_min_square(X2,Z2)
    _1 = noname1[0]
    if answer == '1':
        _2 = noname2[0]
    gamma1 = round(noname1[2],3)
    if answer == '1':
        gamma2 = round(noname2[2],3)
    x1_square = []
    f1_square = []
    x2_square = []
    f2_square = []
    for i in range(len(X1)):
        x1_square.append(_1[i][0])
        f1_square.append(_1[i][2])
        if answer == '1':
            x2_square.append(_2[i][0])
            f2_square.append(_2[i][2])
    plt.plot(x1_square, f1_square, 'r', label=f'Интерполированная \
             функция c G = {gamma1}')
    if answer == '1':
        plt.plot(x2_square, f2_square, 'r', label=f'Интерполированная\
                 функция c G = {gamma2}')
    plt.title("МНК Эйлера-Коши ")
    plt.legend(loc='best', prop={'size': 8}, frameon = False)
    plt.show()
    
    noname1 = newton_here_the_boss(X2,Y2)
    if answer == '1':
        noname2 = newton_here_the_boss(X2,Z2)
    x1_square = []
    f1_square = []
    x2_square = []
    f2_square = []
    for i in range(len(noname1[1])):
        x1_square.append(noname1[0][i])
        f1_square.append(noname1[1][i])
        if answer == '1':
            x2_square.append(noname2[0][i])
            f2_square.append(noname2[1][i])
    plt.plot(x1_square, f1_square, 'r', label=f'Аппроксимированная функция ')
    if answer == '1':
        plt.plot(x2_square, f2_square, 'r', label=f'Аппроксимированная \
                 функция')
    plt.title("Аппроксимация Ньтоном Эйлера-Коши ")
    plt.legend(loc='best', prop={'size': 8}, frameon = False)
    plt.show()
    
    x = []
    y = []
    z = []
    
    noname = rungekutta4(Input)
    for i in range(len(noname)):
        x.append(noname[i][1])
        y.append(noname[i][2])
        if answer == '1':
            z.append(noname[i][3])
    
    plt.plot(x, y, label='Выходные точки Рунге-Кутты функции y(t)')
    if answer == '1':
        plt.plot(x, z, label='Выходные точки Рунге-Кутты функции z(t)')
    plt.title("Выходные точки метода Рунге-Кутты")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    
    plt.show()
    
    noname = method_min_square(x, y)
    _ = noname[0]
    if answer == "1":
        nonamez = method_min_square(x,z)
        _z = nonamez[0]
        gamma_z = round(nonamez[2], 3)
    gamma_y = round(noname[2], 3)
    x_square = []
    y_square = []
    z_square = []
    for i in range(len(x)):
        x_square.append(_[i][0])
        y_square.append(_[i][2])
        if answer == '1':
            z_square.append(_z[i][2])
    
    plt.plot(x_square, y_square, 'r', label=f'МНК y(t) c G = {gamma_y}')
    if answer == '1':
        plt.plot(x_square, z_square, 'b', label=f'МНК z(t) c G = {gamma_z}')
    plt.title("МНК")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    
    plt.show()
    
    noname = newton_here_the_boss(x, y)
    if answer == '1':
        nonamez = newton_here_the_boss(x,z)
    x1_square = []
    f1_square = []
    z_square = []
    for i in range(len(noname[1])):
        x1_square.append(noname[0][i])
        f1_square.append(noname[1][i])
        if answer == '1':
            z_square.append(nonamez[1][i])
    plt.plot(x1_square, f1_square, 'r', label=f'Ньютон y')
    if answer == '1':
        plt.plot(x1_square, z_square, 'b', label=f'Ньютон z')
    plt.title("Ньютон")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    
    plt.show()
    
    delta = []
    dy_left_side_y = diff_left_side(x, y)
    if answer != '1':
        for i in range(len(dy_left_side_y)):
            delta.append(dy_left_side_y[i] - Input[0](x[i], y[i], 1))
        summa_y = sum(delta)
    if answer == '1':
        delta_z = []
        dy_left_side_z = diff_left_side(x, z)
        for i in range(len(dy_left_side_y)):
            delta.append(dy_left_side_y[i] - Input[0](x[i], y[i], z[i]))
            delta_z.append(dy_left_side_z[i] - Input[1](x[i], y[i], z[i]))
        summa_y = sum(delta)
        summa_z = sum(delta_z)
        plt.plot(x[:-1], delta_z, 'r', label=f'Погрешность z sum = {summa_z}')
        plt.legend(loc='best', prop={'size': 8}, frameon=False)
    
    
    plt.plot(x[:-1], delta, 'b', label=f'Погрешность y sum = {summa_y}')
    plt.title("Погрешность")
    plt.legend(loc='best', prop={'size': 8}, frameon=False)
    plt.show()


def Help():
    print('__zadacha1__')
    print('__zadacha1_2__')
    print('__zadacha2__')
    print('__zadacha3_1__')
    print('__zadacha4_1__')
    print('__zadacha4_2__')
    print('__zadacha4_3__')
    print('__zadacha4_4__')
    print('__zadacha4_5__')
    print('__zadacha4_6__')
    print('__zadacha5_1__')
    print('__zadacha5_2__')
    print('__zadacha5_3__')
    print('__zadacha5_4__')
    print('__zadacha5_5__')
    print('__zadacha5_6__')
    print('__zadacha5_7__')
    print('__zadacha6_0__')
    print('__zadacha7__')
    print('__zadacha7_3__')