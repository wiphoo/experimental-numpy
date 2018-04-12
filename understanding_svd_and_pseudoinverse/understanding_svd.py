#!/usr/bin/env python

#	Sigular Value Decomposition (SVD)
#		Trucco, Appendix A.6
#
#	definition
#
#			A = U dot D dot transpose(V)
#
#		A any real m x n matrix
#		U is m x n matrix that columns are eigenvectors of A*transpose(A)
#			A * transpose(A) = U dot D dot transpose(V) dot V dot D dot transpose(U) = U dot D^2 dot transpose(U)
#		V is n x n matrix that columns are eivenvectors of transpose(A)*A 
#			transpose(A)*A = V * D * transpose(U) * U * D * transpose(V) = V * D^2 * transpose(V)
#		D is n x n diagonal matrix (non-negative real values called sigular values)
# 
#	computing the inverse of a matrix using SVD
#		
#		inverse(A) = V dot inverse(D) dot transpose(U)
#			where inverse(D) are 1/singular values
#

#
#		|-     -|
#		| 1 2 1 |
#	A = | 2 3 2 | 
#		| 1 2 1 |
#		|_	   _|
#
#											   |-	     -|
#											   | 6  10 6  |
#	A * transpose( A ) = transpose( A ) * A =  | 10 17 10 |
#											   | 6  10 6  |
#											   |_	     _|

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

#	construct the input array
a = np.array( [ [ 1, 2, 12 ],
				[ 2, 3, 2 ],
				[ 1, 2, 1 ],
			] )
print( 'a = {}'.format( a ) )

print( ' ===############################################################=== ' )
			
# using numpy to find SVD
print( 'calculating the SVD using numpy......................' )
u, s, vt = np.linalg.svd( a )

#	construct a matrix d
d = np.diag( s )

print( '    u = {}'.format( u ) )
print( '    s = {}'.format( s ) )
print( '    vt = {}'.format( vt ) )
print( '    d = {}'.format( d ) )

#	testing the SVD
print( 'testing reconstruct A matrix using SVD' )
reconstructedA = np.dot( u, np.dot( d, vt ) )
assert( np.allclose( a, reconstructedA ) )
print( ' === done === ' )

print( ' ===############################################################=== ' )

print( 'testing construct inverse matrix of A using SVD' )

v = vt.transpose()
inverseD = np.diag( s**-1 )
ut = u.transpose()

print( '    v = {}'.format( v ) )
print( '    inverseD = {}'.format( d ) )
print( '    ut = {}'.format( ut ) )

inverseA = np.dot( v, np.dot( inverseD, ut ) )
print( '    inverseA = {}'.format( inverseA ) )

aMultiplyInverseA = np.matmul( a, inverseA )
print( '    aMultiplyInverseA = {}'.format( aMultiplyInverseA ) )

#	check
assert( np.allclose( aMultiplyInverseA, np.identity( 3 ) ) )
print( ' === done === ' )

