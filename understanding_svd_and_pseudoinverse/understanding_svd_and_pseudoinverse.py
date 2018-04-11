#!/usr/bin/env python

#	Sigular Value Decomposition (SVD)
#		Trucco, Appendix A.6
#
#	definition
#
#			A = U*D*transpose(V)
#
#		A any real m x n matrix
#		U is m x n matrix that columns are eigenvectors of A*transpose(A)
#			A*transpose(A) = U * D * transpose(V) * V * D * transpose(U) = U * D^2 * transpose(U)
#		V is n x n matrix that columns are eivenvectors of transpose(A)*A 
#			transpose(A)*A = V * D * transpose(U) * U * D * transpose(V) = V * D^2 * transpose(V)
#		D is n x n diagonal matrix (non-negative real values called sigular values)
# 
#	computing the inverse of a matrix using SVD
#		
#		inverse(A) = V*inverse(D)*transpose(U)
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

Epsilon = 1e-12


a = np.array( [ [ 1, 2, 1 ],
				[ 2, 3, 2 ],
				[ 1, 2, 1 ],
			] )
print( '  matrix A = {}'.format( a ) )

#	basic inverse matrix
print( '###################################################' )
print( 'basic method inverse matrix...............' )

#	try to inverse using basic method 1/det(a) * adj(a)
try:
	inverseA = np.linalg.inv( a )
	print( '  matrix inverse( A ) = {}'.format( inverseA ) )
except np.linalg.linalg.LinAlgError as e:
	print( 'ERROR!!! Cannot inverse matrix : {}'.format( e ) )

print( '###################################################' )
print( 'SVD method inverse matrix...............' )

#	construct A * transpose( A )
aMultiplyTransposeA = np.array( [ [ 6, 10, 6 ],
									[ 10, 17, 10 ],
									[ 6, 10, 6 ],
								] )
print( '  matrix A * trasnpose( A ) = {}'.format( aMultiplyTransposeA ) )

#	calculate eigenvalues and eigenvectors
print( '      calculating eigen values/vectors' )
eigenvalues, eigenvectors = np.linalg.eig( aMultiplyTransposeA )
print( '      eigenvalues of matrix A * trasnpose( A ) = {}'.format( eigenvalues ) )
print( '      eigenvectors of matrix A * trasnpose( A ) = {}'.format( eigenvectors ) )

#	construct the D matrix from eigenvalues
d = np.array( [ [ eigenvalues[0], 0, 0 ],
				[ 0, eigenvalues[1], 0 ],
				[ 0, 0, eigenvalues[2] ] 
			] )
print( '\n    d matrix = {}'.format( d ) )
			
#	construct the U and V matrix from eigenvectors
u = np.array( eigenvectors )
print( '    u matrix = {}'.format( u ) )

#	construct transpose v
vt = np.array( [ [ eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0] ], 
					[ eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1] ],
					[ eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2] ],
				 ] )
print( '    vt matrix = {}'.format( vt ) )

print( '    testing by calculate A from U*D*transpose(V)..............' )
result = np.matmul( u, d )
result = np.matmul( result, vt )
print( '         result from reconstruct A from SVD = {}'.format( result ) )
assert( ( a[0][0] - result[0][0] < Epsilon ) and ( a[0][1] - result[0][1] < Epsilon ) and ( a[0][2] - result[0][2] < Epsilon )
		and ( a[1][0] - result[1][0] < Epsilon ) and ( a[1][1] - result[1][1] < Epsilon ) and ( a[1][2] - result[1][2] < Epsilon )
		and ( a[2][0] - result[2][0] < Epsilon ) and ( a[2][1] - result[2][1] < Epsilon ) and ( a[2][2] - result[2][2] < Epsilon ) )

print( '..........................................' )
print( '\n\n    preparing data to do a inverse matrix from V*inverse(D)*transpose(U)..............' )

#	calculate inverse of D
inverseD = np.array( [ [ 1./eigenvalues[0] if eigenvalues[0] > Epsilon else 0, 0, 0 ],
						[ 0, 1./eigenvalues[1] if eigenvalues[1] > Epsilon else 0, 0 ],
						[ 0, 0, 1./eigenvalues[2] if eigenvalues[2] > Epsilon else 0 ] 
					] )
print( '\n    inverse(D) matrix = {}'.format( inverseD ) )

#	construct v from eigenvectors
v = np.array( eigenvectors )
print( '    v matrix = {}'.format( v ) )

#	construct transpose u
ut = np.array( [ [ eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0] ], 
					[ eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1] ],
					[ eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2] ],
				 ] )
print( '    ut matrix = {}'.format( ut ) )

print( '\n\n    testing by calculate inverse( A ) from V*inverse(D)*transpose(U)..............' )
inverseA = np.matmul( v, inverseD )
inverseA = np.matmul( result, ut )
print( '         result from reconstruct inverse of A from SVD = {}'.format( inverseA ) )

#	checking the inverse matrix
aMultiplyInverseA = np.matmul( a, inverseA )
print( '         checking the inverse of A by A * inverse(A) = {}'.format( aMultiplyInverseA ) )



			

