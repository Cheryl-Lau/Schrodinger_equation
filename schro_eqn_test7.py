# -*- coding: utf-8 -*-

import numpy as np 
import sympy as sp
import matplotlib.pyplot as plt 

'''
One-dimensional Time-Independent Schrodinger Equation 

Solving TISE in a quantum harmonic oscillator with finite difference method
(forward) on N spatial grid points. Wavefunction = 0 at both boundaries. 
We manually solve the eigenvalues (energies) and their corresponding eigenvectors 
(wavefunctions) with matrix determinant and Gaussian elimination.  

'''


'''
Part 1 - Setting up TISE linear system Hamiltonian matrix 
'''

def init_domain(domain_min, domain_max, num_grid):
    ''' Setting up spatial grid on a 1-D axis to compute TISE solutions '''
    x_grid = np.linspace(domain_min, domain_max, num_grid)
    step_size = np.mean(np.diff(x_grid))
    
    return x_grid, step_size 


def kinetic(matrix_size, step_size):
    ''' Kinetic energy matrix in a system of matrix_size TISE's '''
    # Constant before matrix 
    kinetic_const = -h_bar**2 / (2 * mass * step_size**2)
    
    # Init empty square matrix 
    kinetic_matrix = np.zeros((matrix_size, matrix_size))
    # Construct T matrix 
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i == j:
                kinetic_matrix[i][j] = -2 
            elif i == j+1 and i != 0:
                kinetic_matrix[i][j] = 1 
            elif i == j-1 and j != 0:
                kinetic_matrix[i][j] = 1 
            else:
                kinetic_matrix[i][j] = 0
                
    kinetic_matrix = kinetic_const * kinetic_matrix
    
    return kinetic_matrix 
      

def potential_well():
    ''' Define potential energy as a callable function of spatial axis x in 
    which the system is contained; approximates the quantum harmonic oscillator. 
    '''
    return lambda x: 1/2 * force_const * x**2 
    

def potential(matrix_size, x_grid):
    ''' Potential energy matrix in a system of matrix_size TISE's '''
    # Init function for potential well 
    potential_func = potential_well()
    
    # Init empty square matrix
    potential_matrix = np.zeros((matrix_size, matrix_size))
    # Construct V matrix 
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i == j:
                # potential at current position x 
                potential = potential_func(x_grid[i])
                potential_matrix[i][j] = potential 
            else:
                potential_matrix[i][j] = 0 
                
    return potential_matrix 
                

def hamiltonian(x_grid, step_size):
    ''' Define Hamiltonian matrix of system. H = T + V. 
    Number of wavefunc psi_i in TISE to be solved = N; Provided that psi_0 = 0 
    and psi_N = 0 (from boundary conditions), number of TISE equations to be 
    solved = N-2, hence hamiltonian matrix size = N-2. 
    '''
    matrix_size = num_grid - 2 
    
    kinetic_matrix = kinetic(matrix_size, step_size)
    potential_matrix = potential(matrix_size, x_grid)
    
    hamiltonian_matrix = kinetic_matrix + potential_matrix
    
    return hamiltonian_matrix 


def main_hamiltonian():
    ''' Generating Hamiltonian matrix of system '''    
    x_grid, step_size = init_domain(domain_min, domain_max, num_grid)
    hamiltonian_matrix = hamiltonian(x_grid, step_size)
    
    return hamiltonian_matrix, x_grid


'''
Part 2 -  Solving Eigenvalues 
'''

def hamiltonian_identity(hamiltonian_matrix, X):
    ''' Convert hamiltonian to A-X*I where X is an unknown/eigenval'''
    matrix_size = len(hamiltonian_matrix)
  
    # Indentity matrix multiplied by constant X
    identity_matrix = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i == j:
                identity_matrix[i][j] = 1 
            else:
                identity_matrix[i][j] = 0    
    hamiltonian_identity_matrix = hamiltonian_matrix - X * identity_matrix
    
    return hamiltonian_identity_matrix
 
    
class Node:
    ''' Generates all terms in the matrix determinant expression. 
    Builds an expansion tree with the original matrix as the root;
    in each iteration, expand the nxn matrix into n (n-1)x(n-1) matricies until
    2x2 matricies are obtained, while multiplying all their 'coeffs' (generated
    from all previous branches). 
    Inits obejct and loops method within class to repeatedly generate branches.  
    '''
    def __init__(self, matrix, prev_coeff):
        # init a root node 
        self.matrix = matrix 
        # expand for the first time 
        self.coeff_list, self.minor_matrix_list = expand_matrix(self.matrix)
        # takes coeff passed from previous iteration, make it an obj-attr of the 
        # current matrix
        self.prev_coeff = prev_coeff 
                
    def loop_nodes(self, coeff_list, minor_matrix_list):
        
        for coeff, matrix in zip(self.coeff_list,self.minor_matrix_list):
            
            # For storing and multiplying previous coeff, the coeff product builds 
            # up for each matrix object created
            if self.prev_coeff == None:  # first time for root  
                prev_coeff = coeff   # storing current coeff into prev_coeff for next iter 
            else: # second+ time 
                overall_coeff = coeff * self.prev_coeff                    
                prev_coeff = overall_coeff  # store for next iter 
                               
            # yield the 2x2 matricies and their coeffs (from each previous node)
            if len(matrix) == 2:
                det_output.append((overall_coeff,matrix))

            # init a new object for the newly extracted matrix 
            # taking the coeff of this iteration 
            matrix_obj = Node(matrix, prev_coeff)

            # stop process after returning the 2x2 matricies 
            if len(matrix) > 2:
                # call expand matrix to generate child nodes 
                coeff_list, minor_matrix_list = expand_matrix(matrix)
                
                # call this method 
                matrix_obj.loop_nodes(coeff_list, minor_matrix_list)
            

def expand_matrix(matrix):
    ''' Takes an nxn matrix and expand it into an expression of (n-1)x(n-1) matricies 
    returns the coeff and matrix of minor of individual terms 
    '''
    # take the first row as main 
    top_row = matrix[0]

    coeff_list = []
    minor_matrix_list= []
    # go through each expanded term 
    for col_index, top_val in enumerate(top_row): 
        # Form matrix of minor 
        # all in original matrix apart from those in same row and col of val 
        minor_matrix = np.delete(matrix,0,0)
        minor_matrix = np.delete(minor_matrix,col_index,1)
        
        # coeff of this matrix of minor 
        sign = 1 if col_index %2 == 0 else -1        
        coeff = sign * top_val 

        coeff_list.append(coeff)
        minor_matrix_list.append(minor_matrix)
        
    return coeff_list, minor_matrix_list


def process_det_output(det_output):
    ''' Multiply by the 2x2 coeffs from all previous branches and sum up all terms  
    Calculate determinant 
    '''
    det_sum = 0
    for coeff, matrix in det_output: 
        matrix_det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
        det_sum += coeff * matrix_det
    
    return det_sum


def solve_determinant(matrix):
    ''' Main of Solving matrix determinant '''
    # storing class Node (coeff,matrix) outputs 
    global det_output 
    det_output = []
    
    # init object for root - nxn matrix 
    # no prev_coeff 
    root = Node(matrix, None)    
    # add nodes 
    root.loop_nodes(root.coeff_list, root.minor_matrix_list)
    
    determinant = process_det_output(det_output)   
    
    determinant = sp.simplify(determinant)
    print(determinant)
        
    return determinant 


def solve_eigenval(determinant):
    ''' Find the roots (eigenvalues) of the determinant expression '''
    # Extract polynomial coeffs from det expression 
    det_exp = sp.Poly(determinant)
    coefficients = det_exp.all_coeffs()

    # solve for roots 
    roots = np.roots(coefficients)
    roots = roots.real
    # only positive measurables (schrodinger eigenval)
    eigenval = [val for val in roots if val > 0]
    # remove duplicates 
    final_eigenval = [] 
    [final_eigenval.append(val) for val in eigenval if val not in final_eigenval] 

    return final_eigenval 


def main_eigenval(hamiltonian_matrix):
    ''' Finding eigenvalues of Hamiltonian '''
    # variable for 'lambda' in identity matrix (going to be eigenval)
    X = sp.Symbol('X')
    hamiltonian_identity_matrix = hamiltonian_identity(hamiltonian_matrix, X)

    determinant = solve_determinant(hamiltonian_identity_matrix)    
    eigenvalue_list = solve_eigenval(determinant)  
    
    return eigenvalue_list 


'''
Part 3 -  Solving Eigenvectors for each eigenvalue 
'''

def eigenval_identity(eigenvalue, hamiltonian_matrix):
    ''' Convert matrix to A-X*I | 0 to be passed to gaussian elimination '''
    # A-eigenval*I
    eigenval_identity_matrix = hamiltonian_identity(hamiltonian_matrix, eigenvalue)
    # add a column of zeroes 
    zero_col = np.zeros((len(hamiltonian_matrix),1))
    eigenval_identity_zeros_matrix = np.append(eigenval_identity_matrix, zero_col, axis=1)
    
    return eigenval_identity_zeros_matrix


def elim_2rows(row1, row2):
    ''' Takes two input rows, multiply by the first non-zero element of each other, 
    subtract each other. Output a new row2 to replace the old one.     
    '''
    row1 = list(row1)  # convert 
    row2 = list(row2)
    # locate the first non-zero element 
    nz_index = row1.index([num for num in row1 if num != 0][0])
    
    temp_row1 = row1
    # cross multiply 
    row1 = [r1*row2[nz_index] for r1 in row1]
    row2 = [r2*temp_row1[nz_index] for r2 in row2]
    # subtract and return updated row2 
    row2 = np.array(row1) - np.array(row2)
    
    return row2


def count_head_zeroes(row):
    ''' checks number of zeros at the beginning of a given row '''
    # index of first non-zero element 
    nz_index = []
    for i, num in enumerate(row):
        if num != 0:
            nz_index.append(i)
    head_zeroes = nz_index[0]  # number of zeros = first non-zero index
    
    return head_zeroes


def count_zeroes(row):
    ''' checks total number of zeros in a given row '''
    return len([num for num in row if num == 0])
    

def check_terminate(matrix, num_col):
    ''' checks through entire matrix to check if the elimination can be terminated '''
    terminate = False  # init 
    for row in matrix:
        # number of non-zero elements in row
        num_nz = num_col - count_zeroes(row)
        # system is solvable when only 2 numbers left 
        if num_nz == 2:  
            terminate = True 
        # no solution if 1/0 numbers left 
        if num_nz == 1 or num_nz == 0:
            terminate = True 
            raise Exception('System has no unique solution')
            
    return terminate   # boolean 


def same_zeroes_rows(row_head_zeroes, curr_col):
    ''' checks if the number of head zeroes in the given row matched current column '''
    same_zeroes_rows_index = []
    for i, zeroes in enumerate(row_head_zeroes):
        if zeroes == curr_col:
            same_zeroes_rows_index.append(i)
            
    return same_zeroes_rows_index


def choose_row(row_head_zeroes, curr_col):
    ''' Pick two rows with same number of head zeroes before the current column 
    (where num of zeroes in both = current column index);
    Otherwise, when no two rows with same num of zeroes before this column is left, 
    shift to the next column. 
    '''   
    same_zeroes_rows_index = same_zeroes_rows(row_head_zeroes, curr_col)
             
    # if insufficient rows have same number of zeroes before this column 
    while len(same_zeroes_rows_index) < 2:
        curr_col += 1  # shift column 
        same_zeroes_rows_index = same_zeroes_rows(row_head_zeroes, curr_col)

    # pick first and last row 
    row1_index = same_zeroes_rows_index[0]
    row2_index = same_zeroes_rows_index[-1]
    
    return row1_index, row2_index, curr_col


def gauss_elim(matrix, num_row, num_col):
    ''' Main process of gaussian elimination '''
    terminate = False  # init 
    curr_col = 0   # init current column being dealt with 
    
    while terminate == False: 
        
        # evaluate number of head zeroes in each row of updated matrix 
        row_head_zeroes = [count_head_zeroes(row) for row in matrix]

        # pick two rows; update shifted curr_col 
        row1_index, row2_index, curr_col = choose_row(row_head_zeroes, curr_col)
        row1 = matrix[row1_index][:]
        row2 = matrix[row2_index][:]
        # eliminate 
        row2 = elim_2rows(row1, row2)
        
        # Update row2 in matrix 
        matrix[row2_index] = row2 

        # Repeat process until system becomes solvable 
        terminate = check_terminate(matrix, num_col)
        
    return matrix


def solve_eigenvec(matrix, num_row, num_col):
    ''' Solves x1, x2, ..., xn in eigenvector with final output matrix 
    '''
    # sort rows by biggest term 
    row_index = [i for i in range(num_row)]
    num_digits = [num_col - count_head_zeroes(row) for row in matrix]
    sorted_num_digits = sorted(zip(row_index,num_digits), reverse=True)

    # init empty array for storing eigenvector solutions 
    eigenvec = [0] * (num_col-1)
    
    # go through each row in matrix 
    for row_index_num_digits in sorted_num_digits:
        
        row_index = row_index_num_digits[0]
        row = matrix[row_index]

        # solution of this row 
        sol = matrix[row_index][-1]
        
        # detect col indicies of non-zero digits in this row (excluding the solution)
        col_nz_index = np.delete(np.where(row != 0),-1)

        # rearrage eqn to solve for eigenvec of largest term 
        largest_term = sol
        for col_index in col_nz_index:
            # apply eigenvec values found from previous iteration 
            largest_term -= matrix[row_index][col_index] * eigenvec[col_index]            
        largest_term /= matrix[row_index][col_nz_index[0]]
        
        # store result into eigenvector array 
        eigenvec[col_nz_index[0]] = largest_term 
          
    return eigenvec 


def main_eigenvec(eigenvalue, hamiltonian_matrix):
    ''' Finding eigenvector of a given eigenvalue '''
    matrix = eigenval_identity(eigenvalue, hamiltonian_matrix)

    num_row = len(matrix)
    num_col = len(matrix[0][:])
    
    # gaussian elimination 
    final_matrix = gauss_elim(matrix, num_row, num_col)    
    # solve for eigenvector 
    eigenvec = solve_eigenvec(final_matrix, num_row, num_col)
    
    return eigenvec 


'''
Part 4 - Plotting states  
'''

def plot_state(x_grid, eigenvec_list):
    ''' Plot wavefunctions of a few quantum states '''
    eigenvec = eigenvec_list[0:num_states]
    
    for i in range(len(eigenvec)):
        plt.plot(x_grid, eigenvec[i], label="{}".format(i))
    plt.show()




def main():
    ''' Full calculation process '''
    # Part 1 Setting up matrix 
    hamiltonian_matrix, x_grid = main_hamiltonian()
    # Part 2 Eigenvalues 
    eigenval_list = main_eigenval(hamiltonian_matrix)
    # Part 3 Eigenvectors 
    eigenvec_list = []
    for eigenval in eigenval_list:
        eigenvec = main_eigenvec(eigenval, hamiltonian_matrix)
        # Taking psi_i at BC into account 
        eigenvec.insert(0,0)
        eigenvec.insert(len(eigenvec),0)
        eigenvec_list.append(eigenvec)
        
    print(np.shape(eigenvec_list))
    
    plot_state(x_grid, eigenvec_list)
 
   
'''
Input Settings 
'''    

# Planck constant 
h_bar = 6.62607015e-34

# Bond force constant of HCl molecule 
force_const = 480

# Hydrogen atom mass 
mass = 1.67355755e-27

# Set spatial domain
domain_min = -10
domain_max = 10
# Number of points in grid, i.e. total number of psi_i, N 
num_grid = 100 


# Number of quantum states to plot 
num_states = 5 
    


if __name__ == '__main__':
    main()








































