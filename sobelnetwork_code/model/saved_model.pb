??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??	
v
Adam/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/Variable/v
o
#Adam/Variable/v/Read/ReadVariableOpReadVariableOpAdam/Variable/v*
_output_shapes
:*
dtype0
~
Adam/Variable/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/v_1
w
%Adam/Variable/v_1/Read/ReadVariableOpReadVariableOpAdam/Variable/v_1*
_output_shapes

:*
dtype0
z
Adam/Variable/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/v_2
s
%Adam/Variable/v_2/Read/ReadVariableOpReadVariableOpAdam/Variable/v_2*
_output_shapes
:*
dtype0

Adam/Variable/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*"
shared_nameAdam/Variable/v_3
x
%Adam/Variable/v_3/Read/ReadVariableOpReadVariableOpAdam/Variable/v_3*
_output_shapes
:	?*
dtype0
{
Adam/Variable/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/Variable/v_4
t
%Adam/Variable/v_4/Read/ReadVariableOpReadVariableOpAdam/Variable/v_4*
_output_shapes	
:?*
dtype0

Adam/Variable/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:	=?*"
shared_nameAdam/Variable/v_5
x
%Adam/Variable/v_5/Read/ReadVariableOpReadVariableOpAdam/Variable/v_5*
_output_shapes
:	=?*
dtype0
z
Adam/Variable/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:=*"
shared_nameAdam/Variable/v_6
s
%Adam/Variable/v_6/Read/ReadVariableOpReadVariableOpAdam/Variable/v_6*
_output_shapes
:=*
dtype0

Adam/Variable/v_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?=*"
shared_nameAdam/Variable/v_7
x
%Adam/Variable/v_7/Read/ReadVariableOpReadVariableOpAdam/Variable/v_7*
_output_shapes
:	?=*
dtype0
{
Adam/Variable/v_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/Variable/v_8
t
%Adam/Variable/v_8/Read/ReadVariableOpReadVariableOpAdam/Variable/v_8*
_output_shapes	
:?*
dtype0

Adam/Variable/v_9VarHandleOp*
_output_shapes
: *
dtype0*
shape:	y?*"
shared_nameAdam/Variable/v_9
x
%Adam/Variable/v_9/Read/ReadVariableOpReadVariableOpAdam/Variable/v_9*
_output_shapes
:	y?*
dtype0
|
Adam/Variable/v_10VarHandleOp*
_output_shapes
: *
dtype0*
shape:y*#
shared_nameAdam/Variable/v_10
u
&Adam/Variable/v_10/Read/ReadVariableOpReadVariableOpAdam/Variable/v_10*
_output_shapes
:y*
dtype0
?
Adam/Variable/v_11VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ty*#
shared_nameAdam/Variable/v_11
y
&Adam/Variable/v_11/Read/ReadVariableOpReadVariableOpAdam/Variable/v_11*
_output_shapes

:ty*
dtype0
|
Adam/Variable/v_12VarHandleOp*
_output_shapes
: *
dtype0*
shape:t*#
shared_nameAdam/Variable/v_12
u
&Adam/Variable/v_12/Read/ReadVariableOpReadVariableOpAdam/Variable/v_12*
_output_shapes
:t*
dtype0
?
Adam/Variable/v_13VarHandleOp*
_output_shapes
: *
dtype0*
shape
:t*#
shared_nameAdam/Variable/v_13
y
&Adam/Variable/v_13/Read/ReadVariableOpReadVariableOpAdam/Variable/v_13*
_output_shapes

:t*
dtype0
v
Adam/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/Variable/m
o
#Adam/Variable/m/Read/ReadVariableOpReadVariableOpAdam/Variable/m*
_output_shapes
:*
dtype0
~
Adam/Variable/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/m_1
w
%Adam/Variable/m_1/Read/ReadVariableOpReadVariableOpAdam/Variable/m_1*
_output_shapes

:*
dtype0
z
Adam/Variable/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/m_2
s
%Adam/Variable/m_2/Read/ReadVariableOpReadVariableOpAdam/Variable/m_2*
_output_shapes
:*
dtype0

Adam/Variable/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*"
shared_nameAdam/Variable/m_3
x
%Adam/Variable/m_3/Read/ReadVariableOpReadVariableOpAdam/Variable/m_3*
_output_shapes
:	?*
dtype0
{
Adam/Variable/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/Variable/m_4
t
%Adam/Variable/m_4/Read/ReadVariableOpReadVariableOpAdam/Variable/m_4*
_output_shapes	
:?*
dtype0

Adam/Variable/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:	=?*"
shared_nameAdam/Variable/m_5
x
%Adam/Variable/m_5/Read/ReadVariableOpReadVariableOpAdam/Variable/m_5*
_output_shapes
:	=?*
dtype0
z
Adam/Variable/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:=*"
shared_nameAdam/Variable/m_6
s
%Adam/Variable/m_6/Read/ReadVariableOpReadVariableOpAdam/Variable/m_6*
_output_shapes
:=*
dtype0

Adam/Variable/m_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?=*"
shared_nameAdam/Variable/m_7
x
%Adam/Variable/m_7/Read/ReadVariableOpReadVariableOpAdam/Variable/m_7*
_output_shapes
:	?=*
dtype0
{
Adam/Variable/m_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/Variable/m_8
t
%Adam/Variable/m_8/Read/ReadVariableOpReadVariableOpAdam/Variable/m_8*
_output_shapes	
:?*
dtype0

Adam/Variable/m_9VarHandleOp*
_output_shapes
: *
dtype0*
shape:	y?*"
shared_nameAdam/Variable/m_9
x
%Adam/Variable/m_9/Read/ReadVariableOpReadVariableOpAdam/Variable/m_9*
_output_shapes
:	y?*
dtype0
|
Adam/Variable/m_10VarHandleOp*
_output_shapes
: *
dtype0*
shape:y*#
shared_nameAdam/Variable/m_10
u
&Adam/Variable/m_10/Read/ReadVariableOpReadVariableOpAdam/Variable/m_10*
_output_shapes
:y*
dtype0
?
Adam/Variable/m_11VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ty*#
shared_nameAdam/Variable/m_11
y
&Adam/Variable/m_11/Read/ReadVariableOpReadVariableOpAdam/Variable/m_11*
_output_shapes

:ty*
dtype0
|
Adam/Variable/m_12VarHandleOp*
_output_shapes
: *
dtype0*
shape:t*#
shared_nameAdam/Variable/m_12
u
&Adam/Variable/m_12/Read/ReadVariableOpReadVariableOpAdam/Variable/m_12*
_output_shapes
:t*
dtype0
?
Adam/Variable/m_13VarHandleOp*
_output_shapes
: *
dtype0*
shape
:t*#
shared_nameAdam/Variable/m_13
y
&Adam/Variable/m_13/Read/ReadVariableOpReadVariableOpAdam/Variable/m_13*
_output_shapes

:t*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
h
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
p

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable_1
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:*
dtype0
l

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_2
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0
q

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name
Variable_3
j
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:	?*
dtype0
m

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Variable_4
f
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes	
:?*
dtype0
q

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:	=?*
shared_name
Variable_5
j
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:	=?*
dtype0
l

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:=*
shared_name
Variable_6
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:=*
dtype0
q

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?=*
shared_name
Variable_7
j
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:	?=*
dtype0
m

Variable_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Variable_8
f
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes	
:?*
dtype0
q

Variable_9VarHandleOp*
_output_shapes
: *
dtype0*
shape:	y?*
shared_name
Variable_9
j
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:	y?*
dtype0
n
Variable_10VarHandleOp*
_output_shapes
: *
dtype0*
shape:y*
shared_nameVariable_10
g
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:y*
dtype0
r
Variable_11VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ty*
shared_nameVariable_11
k
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes

:ty*
dtype0
n
Variable_12VarHandleOp*
_output_shapes
: *
dtype0*
shape:t*
shared_nameVariable_12
g
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:t*
dtype0
r
Variable_13VarHandleOp*
_output_shapes
: *
dtype0*
shape
:t*
shared_nameVariable_13
k
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes

:t*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_37062

NoOpNoOp
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?P
value?PB?P B?P
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
W1
	b1

dp1
W2
b2
dp2
W3
b3
dp3
W4
b4
dp4
W5
b5
dp5
W6
b6
dp6
W7
b7
w
	optimizer

signatures*
j
0
	1
2
3
4
5
6
7
8
9
10
11
12
13*
j
0
	1
2
3
4
5
6
7
8
9
10
11
12
13*
* 
?
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
$trace_0
%trace_1
&trace_2
'trace_3* 
6
(trace_0
)trace_1
*trace_2
+trace_3* 
* 
B<
VARIABLE_VALUEVariable_13W1/.ATTRIBUTES/VARIABLE_VALUE*
B<
VARIABLE_VALUEVariable_12b1/.ATTRIBUTES/VARIABLE_VALUE*
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator* 
B<
VARIABLE_VALUEVariable_11W2/.ATTRIBUTES/VARIABLE_VALUE*
B<
VARIABLE_VALUEVariable_10b2/.ATTRIBUTES/VARIABLE_VALUE*
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator* 
A;
VARIABLE_VALUE
Variable_9W3/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_8b3/.ATTRIBUTES/VARIABLE_VALUE*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@_random_generator* 
A;
VARIABLE_VALUE
Variable_7W4/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_6b4/.ATTRIBUTES/VARIABLE_VALUE*
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator* 
A;
VARIABLE_VALUE
Variable_5W5/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_4b5/.ATTRIBUTES/VARIABLE_VALUE*
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_random_generator* 
A;
VARIABLE_VALUE
Variable_3W6/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUE
Variable_2b6/.ATTRIBUTES/VARIABLE_VALUE*
(
O	keras_api
P_random_generator* 
A;
VARIABLE_VALUE
Variable_1W7/.ATTRIBUTES/VARIABLE_VALUE*
?9
VARIABLE_VALUEVariableb7/.ATTRIBUTES/VARIABLE_VALUE*
5
Q0
R1
S2
T3
U4
V5
W6*
?
Xiter

Ybeta_1

Zbeta_2
	[decay
\learning_ratem?	m?m?m?m?m?m?m?m?m?m?m?m?m?v?	v?v?v?v?v?v?v?v?v?v?v?v?v?*

]serving_default* 
* 
,

0
1
2
3
4
5* 
'
^0
_1
`2
a3
b4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

htrace_0
itrace_1* 

jtrace_0
ktrace_1* 
* 
* 
* 
* 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

qtrace_0
rtrace_1* 

strace_0
ttrace_1* 
* 
* 
* 
* 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

ztrace_0
{trace_1* 

|trace_0
}trace_1* 
* 
* 
* 
* 
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 

0
	1*

0
1*

0
1*

0
1*

0
1*

0
1*

0
1*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
e_
VARIABLE_VALUEAdam/Variable/m_139W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/Variable/m_129b1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/Variable/m_119W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/Variable/m_109b2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_99W3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_89b3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_79W4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_69b4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_59W5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_49b5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_39W6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_29b6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/m_19W7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/Variable/m9b7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/Variable/v_139W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/Variable/v_129b1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/Variable/v_119W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/Variable/v_109b2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_99W3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_89b3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_79W4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_69b4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_59W5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_49b5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_39W6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_29b6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/Variable/v_19W7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/Variable/v9b7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_13/Read/ReadVariableOpVariable_12/Read/ReadVariableOpVariable_11/Read/ReadVariableOpVariable_10/Read/ReadVariableOpVariable_9/Read/ReadVariableOpVariable_8/Read/ReadVariableOpVariable_7/Read/ReadVariableOpVariable_6/Read/ReadVariableOpVariable_5/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp&Adam/Variable/m_13/Read/ReadVariableOp&Adam/Variable/m_12/Read/ReadVariableOp&Adam/Variable/m_11/Read/ReadVariableOp&Adam/Variable/m_10/Read/ReadVariableOp%Adam/Variable/m_9/Read/ReadVariableOp%Adam/Variable/m_8/Read/ReadVariableOp%Adam/Variable/m_7/Read/ReadVariableOp%Adam/Variable/m_6/Read/ReadVariableOp%Adam/Variable/m_5/Read/ReadVariableOp%Adam/Variable/m_4/Read/ReadVariableOp%Adam/Variable/m_3/Read/ReadVariableOp%Adam/Variable/m_2/Read/ReadVariableOp%Adam/Variable/m_1/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp&Adam/Variable/v_13/Read/ReadVariableOp&Adam/Variable/v_12/Read/ReadVariableOp&Adam/Variable/v_11/Read/ReadVariableOp&Adam/Variable/v_10/Read/ReadVariableOp%Adam/Variable/v_9/Read/ReadVariableOp%Adam/Variable/v_8/Read/ReadVariableOp%Adam/Variable/v_7/Read/ReadVariableOp%Adam/Variable/v_6/Read/ReadVariableOp%Adam/Variable/v_5/Read/ReadVariableOp%Adam/Variable/v_4/Read/ReadVariableOp%Adam/Variable/v_3/Read/ReadVariableOp%Adam/Variable/v_2/Read/ReadVariableOp%Adam/Variable/v_1/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_37606
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_4count_4total_3count_3total_2count_2total_1count_1totalcountAdam/Variable/m_13Adam/Variable/m_12Adam/Variable/m_11Adam/Variable/m_10Adam/Variable/m_9Adam/Variable/m_8Adam/Variable/m_7Adam/Variable/m_6Adam/Variable/m_5Adam/Variable/m_4Adam/Variable/m_3Adam/Variable/m_2Adam/Variable/m_1Adam/Variable/mAdam/Variable/v_13Adam/Variable/v_12Adam/Variable/v_11Adam/Variable/v_10Adam/Variable/v_9Adam/Variable/v_8Adam/Variable/v_7Adam/Variable/v_6Adam/Variable/v_5Adam/Variable/v_4Adam/Variable/v_3Adam/Variable/v_2Adam/Variable/v_1Adam/Variable/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_37787??
?
?
#__inference_signature_wrapper_37062
input_1
unknown:t
	unknown_0:t
	unknown_1:ty
	unknown_2:y
	unknown_3:	y?
	unknown_4:	?
	unknown_5:	?=
	unknown_6:=
	unknown_7:	=?
	unknown_8:	?
	unknown_9:	?

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_36514o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_36533

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????t[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????t"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????t:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_37319

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????y[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????y"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????y:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?
b
)__inference_dropout_3_layer_call_fn_37368

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36679o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????=22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????=
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_37287

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_36748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????t`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????t22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?
?
/__inference_sobolev_network_layer_call_fn_36636
input_1
unknown:t
	unknown_0:t
	unknown_1:ty
	unknown_2:y
	unknown_3:	y?
	unknown_4:	?
	unknown_5:	?=
	unknown_6:=
	unknown_7:	=?
	unknown_8:	?
	unknown_9:	?

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36605o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
/__inference_sobolev_network_layer_call_fn_37128
x
unknown:t
	unknown_0:t
	unknown_1:ty
	unknown_2:y
	unknown_3:	y?
	unknown_4:	?
	unknown_5:	?=
	unknown_6:=
	unknown_7:	=?
	unknown_8:	?
	unknown_9:	?

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_nameX
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_37373

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????=[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????="!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????=:O K
'
_output_shapes
:?????????=
 
_user_specified_nameinputs
?E
?

J__inference_sobolev_network_layer_call_and_return_conditional_losses_36843
x0
matmul_readvariableop_resource:t)
add_readvariableop_resource:t2
 matmul_1_readvariableop_resource:ty+
add_1_readvariableop_resource:y3
 matmul_2_readvariableop_resource:	y?,
add_2_readvariableop_resource:	?3
 matmul_3_readvariableop_resource:	?=+
add_3_readvariableop_resource:=3
 matmul_4_readvariableop_resource:	=?,
add_4_readvariableop_resource:	?3
 matmul_5_readvariableop_resource:	?+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?add/ReadVariableOp?add_1/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOp?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCallt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:t*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????t?
dropout/StatefulPartitionedCallStatefulPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_36748x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ty*
dtype0?
MatMul_1MatMul(dropout/StatefulPartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:y*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????y?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall
Tanh_1:y:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_36725y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	y?*
dtype0?
MatMul_2MatMul*dropout_1/StatefulPartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????P
SigmoidSigmoid	add_2:z:0*
T0*(
_output_shapes
:???????????
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallSigmoid:y:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_36702y
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?=*
dtype0?
MatMul_3MatMul*dropout_2/StatefulPartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:=*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=Q
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:?????????=?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallSigmoid_1:y:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36679y
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes
:	=?*
dtype0?
MatMul_4MatMul*dropout_3/StatefulPartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????K
	LeakyRelu	LeakyRelu	add_4:z:0*(
_output_shapes
:???????????
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_36656y
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMul_5MatMul*dropout_4/StatefulPartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:?????????x
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_nameX
?
E
)__inference_dropout_3_layer_call_fn_37363

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36575`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????=:O K
'
_output_shapes
:?????????=
 
_user_specified_nameinputs
?
E
)__inference_dropout_4_layer_call_fn_37390

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_36589a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_36702

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_36656

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?>
?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36605
x0
matmul_readvariableop_resource:t)
add_readvariableop_resource:t2
 matmul_1_readvariableop_resource:ty+
add_1_readvariableop_resource:y3
 matmul_2_readvariableop_resource:	y?,
add_2_readvariableop_resource:	?3
 matmul_3_readvariableop_resource:	?=+
add_3_readvariableop_resource:=3
 matmul_4_readvariableop_resource:	=?,
add_4_readvariableop_resource:	?3
 matmul_5_readvariableop_resource:	?+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?add/ReadVariableOp?add_1/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:t*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????t?
dropout/PartitionedCallPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_36533x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ty*
dtype0?
MatMul_1MatMul dropout/PartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:y*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????y?
dropout_1/PartitionedCallPartitionedCall
Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_36547y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	y?*
dtype0?
MatMul_2MatMul"dropout_1/PartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????P
SigmoidSigmoid	add_2:z:0*
T0*(
_output_shapes
:???????????
dropout_2/PartitionedCallPartitionedCallSigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_36561y
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?=*
dtype0?
MatMul_3MatMul"dropout_2/PartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:=*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=Q
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:?????????=?
dropout_3/PartitionedCallPartitionedCallSigmoid_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36575y
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes
:	=?*
dtype0?
MatMul_4MatMul"dropout_3/PartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????K
	LeakyRelu	LeakyRelu	add_4:z:0*(
_output_shapes
:???????????
dropout_4/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_36589y
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMul_5MatMul"dropout_4/PartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:?????????x
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_nameX
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_36575

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????=[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????="!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????=:O K
'
_output_shapes
:?????????=
 
_user_specified_nameinputs
?	
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_37385

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????=C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????=*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????=o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????=i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????=Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????=:O K
'
_output_shapes
:?????????=
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_37336

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_36561a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_37400

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?F
?

J__inference_sobolev_network_layer_call_and_return_conditional_losses_37021
input_10
matmul_readvariableop_resource:t)
add_readvariableop_resource:t2
 matmul_1_readvariableop_resource:ty+
add_1_readvariableop_resource:y3
 matmul_2_readvariableop_resource:	y?,
add_2_readvariableop_resource:	?3
 matmul_3_readvariableop_resource:	?=+
add_3_readvariableop_resource:=3
 matmul_4_readvariableop_resource:	=?,
add_4_readvariableop_resource:	?3
 matmul_5_readvariableop_resource:	?+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?add/ReadVariableOp?add_1/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOp?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCallt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t*
dtype0j
MatMulMatMulinput_1MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:t*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????t?
dropout/StatefulPartitionedCallStatefulPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_36748x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ty*
dtype0?
MatMul_1MatMul(dropout/StatefulPartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:y*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????y?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall
Tanh_1:y:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_36725y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	y?*
dtype0?
MatMul_2MatMul*dropout_1/StatefulPartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????P
SigmoidSigmoid	add_2:z:0*
T0*(
_output_shapes
:???????????
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallSigmoid:y:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_36702y
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?=*
dtype0?
MatMul_3MatMul*dropout_2/StatefulPartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:=*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=Q
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:?????????=?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallSigmoid_1:y:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36679y
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes
:	=?*
dtype0?
MatMul_4MatMul*dropout_3/StatefulPartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????K
	LeakyRelu	LeakyRelu	add_4:z:0*(
_output_shapes
:???????????
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_36656y
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMul_5MatMul*dropout_4/StatefulPartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:?????????x
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_36589

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_37314

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_36725o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????y22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?	
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_36679

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????=C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????=*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????=o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????=i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????=Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????="
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????=:O K
'
_output_shapes
:?????????=
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_37346

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_37309

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_36547`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????y"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????y:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?
?
/__inference_sobolev_network_layer_call_fn_36907
input_1
unknown:t
	unknown_0:t
	unknown_1:ty
	unknown_2:y
	unknown_3:	y?
	unknown_4:	?
	unknown_5:	?=
	unknown_6:=
	unknown_7:	=?
	unknown_8:	?
	unknown_9:	?

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
/__inference_sobolev_network_layer_call_fn_37095
x
unknown:t
	unknown_0:t
	unknown_1:ty
	unknown_2:y
	unknown_3:	y?
	unknown_4:	?
	unknown_5:	?=
	unknown_6:=
	unknown_7:	=?
	unknown_8:	?
	unknown_9:	?

unknown_10:

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36605o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????

_user_specified_nameX
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_36547

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????y[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????y"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????y:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_37304

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????tC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????t*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????to
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????ti
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????tY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????t"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????t:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_36725

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????yC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????y*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????yo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????yi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????yY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????y"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????y:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?5
?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37185
x0
matmul_readvariableop_resource:t)
add_readvariableop_resource:t2
 matmul_1_readvariableop_resource:ty+
add_1_readvariableop_resource:y3
 matmul_2_readvariableop_resource:	y?,
add_2_readvariableop_resource:	?3
 matmul_3_readvariableop_resource:	?=+
add_3_readvariableop_resource:=3
 matmul_4_readvariableop_resource:	=?,
add_4_readvariableop_resource:	?3
 matmul_5_readvariableop_resource:	?+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?add/ReadVariableOp?add_1/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:t*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????tX
dropout/IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:?????????tx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ty*
dtype0?
MatMul_1MatMuldropout/Identity:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:y*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????y\
dropout_1/IdentityIdentity
Tanh_1:y:0*
T0*'
_output_shapes
:?????????yy
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	y?*
dtype0?
MatMul_2MatMuldropout_1/Identity:output:0MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????P
SigmoidSigmoid	add_2:z:0*
T0*(
_output_shapes
:??????????^
dropout_2/IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:??????????y
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?=*
dtype0?
MatMul_3MatMuldropout_2/Identity:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:=*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=Q
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:?????????=_
dropout_3/IdentityIdentitySigmoid_1:y:0*
T0*'
_output_shapes
:?????????=y
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes
:	=?*
dtype0?
MatMul_4MatMuldropout_3/Identity:output:0MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????K
	LeakyRelu	LeakyRelu	add_4:z:0*(
_output_shapes
:??????????j
dropout_4/IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????y
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMul_5MatMuldropout_4/Identity:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:?????????x
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_nameX
?b
?
__inference__traced_save_37606
file_prefix*
&savev2_variable_13_read_readvariableop*
&savev2_variable_12_read_readvariableop*
&savev2_variable_11_read_readvariableop*
&savev2_variable_10_read_readvariableop)
%savev2_variable_9_read_readvariableop)
%savev2_variable_8_read_readvariableop)
%savev2_variable_7_read_readvariableop)
%savev2_variable_6_read_readvariableop)
%savev2_variable_5_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_1_read_readvariableop'
#savev2_variable_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop1
-savev2_adam_variable_m_13_read_readvariableop1
-savev2_adam_variable_m_12_read_readvariableop1
-savev2_adam_variable_m_11_read_readvariableop1
-savev2_adam_variable_m_10_read_readvariableop0
,savev2_adam_variable_m_9_read_readvariableop0
,savev2_adam_variable_m_8_read_readvariableop0
,savev2_adam_variable_m_7_read_readvariableop0
,savev2_adam_variable_m_6_read_readvariableop0
,savev2_adam_variable_m_5_read_readvariableop0
,savev2_adam_variable_m_4_read_readvariableop0
,savev2_adam_variable_m_3_read_readvariableop0
,savev2_adam_variable_m_2_read_readvariableop0
,savev2_adam_variable_m_1_read_readvariableop.
*savev2_adam_variable_m_read_readvariableop1
-savev2_adam_variable_v_13_read_readvariableop1
-savev2_adam_variable_v_12_read_readvariableop1
-savev2_adam_variable_v_11_read_readvariableop1
-savev2_adam_variable_v_10_read_readvariableop0
,savev2_adam_variable_v_9_read_readvariableop0
,savev2_adam_variable_v_8_read_readvariableop0
,savev2_adam_variable_v_7_read_readvariableop0
,savev2_adam_variable_v_6_read_readvariableop0
,savev2_adam_variable_v_5_read_readvariableop0
,savev2_adam_variable_v_4_read_readvariableop0
,savev2_adam_variable_v_3_read_readvariableop0
,savev2_adam_variable_v_2_read_readvariableop0
,savev2_adam_variable_v_1_read_readvariableop.
*savev2_adam_variable_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:BW1/.ATTRIBUTES/VARIABLE_VALUEBb1/.ATTRIBUTES/VARIABLE_VALUEBW2/.ATTRIBUTES/VARIABLE_VALUEBb2/.ATTRIBUTES/VARIABLE_VALUEBW3/.ATTRIBUTES/VARIABLE_VALUEBb3/.ATTRIBUTES/VARIABLE_VALUEBW4/.ATTRIBUTES/VARIABLE_VALUEBb4/.ATTRIBUTES/VARIABLE_VALUEBW5/.ATTRIBUTES/VARIABLE_VALUEBb5/.ATTRIBUTES/VARIABLE_VALUEBW6/.ATTRIBUTES/VARIABLE_VALUEBb6/.ATTRIBUTES/VARIABLE_VALUEBW7/.ATTRIBUTES/VARIABLE_VALUEBb7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_variable_13_read_readvariableop&savev2_variable_12_read_readvariableop&savev2_variable_11_read_readvariableop&savev2_variable_10_read_readvariableop%savev2_variable_9_read_readvariableop%savev2_variable_8_read_readvariableop%savev2_variable_7_read_readvariableop%savev2_variable_6_read_readvariableop%savev2_variable_5_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_1_read_readvariableop#savev2_variable_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop-savev2_adam_variable_m_13_read_readvariableop-savev2_adam_variable_m_12_read_readvariableop-savev2_adam_variable_m_11_read_readvariableop-savev2_adam_variable_m_10_read_readvariableop,savev2_adam_variable_m_9_read_readvariableop,savev2_adam_variable_m_8_read_readvariableop,savev2_adam_variable_m_7_read_readvariableop,savev2_adam_variable_m_6_read_readvariableop,savev2_adam_variable_m_5_read_readvariableop,savev2_adam_variable_m_4_read_readvariableop,savev2_adam_variable_m_3_read_readvariableop,savev2_adam_variable_m_2_read_readvariableop,savev2_adam_variable_m_1_read_readvariableop*savev2_adam_variable_m_read_readvariableop-savev2_adam_variable_v_13_read_readvariableop-savev2_adam_variable_v_12_read_readvariableop-savev2_adam_variable_v_11_read_readvariableop-savev2_adam_variable_v_10_read_readvariableop,savev2_adam_variable_v_9_read_readvariableop,savev2_adam_variable_v_8_read_readvariableop,savev2_adam_variable_v_7_read_readvariableop,savev2_adam_variable_v_6_read_readvariableop,savev2_adam_variable_v_5_read_readvariableop,savev2_adam_variable_v_4_read_readvariableop,savev2_adam_variable_v_3_read_readvariableop,savev2_adam_variable_v_2_read_readvariableop,savev2_adam_variable_v_1_read_readvariableop*savev2_adam_variable_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :t:t:ty:y:	y?:?:	?=:=:	=?:?:	?:::: : : : : : : : : : : : : : : :t:t:ty:y:	y?:?:	?=:=:	=?:?:	?::::t:t:ty:y:	y?:?:	?=:=:	=?:?:	?:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:t: 

_output_shapes
:t:$ 

_output_shapes

:ty: 

_output_shapes
:y:%!

_output_shapes
:	y?:!

_output_shapes	
:?:%!

_output_shapes
:	?=: 

_output_shapes
:=:%	!

_output_shapes
:	=?:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:t: 

_output_shapes
:t:$  

_output_shapes

:ty: !

_output_shapes
:y:%"!

_output_shapes
:	y?:!#

_output_shapes	
:?:%$!

_output_shapes
:	?=: %

_output_shapes
:=:%&!

_output_shapes
:	=?:!'

_output_shapes	
:?:%(!

_output_shapes
:	?: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:t: -

_output_shapes
:t:$. 

_output_shapes

:ty: /

_output_shapes
:y:%0!

_output_shapes
:	y?:!1

_output_shapes	
:?:%2!

_output_shapes
:	?=: 3

_output_shapes
:=:%4!

_output_shapes
:	=?:!5

_output_shapes	
:?:%6!

_output_shapes
:	?: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
:::

_output_shapes
: 
?
b
)__inference_dropout_4_layer_call_fn_37395

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_36656p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_37341

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_36702p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_36561

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_37282

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_36533`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????t"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????t:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?	
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_37412

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?Z
?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37277
x0
matmul_readvariableop_resource:t)
add_readvariableop_resource:t2
 matmul_1_readvariableop_resource:ty+
add_1_readvariableop_resource:y3
 matmul_2_readvariableop_resource:	y?,
add_2_readvariableop_resource:	?3
 matmul_3_readvariableop_resource:	?=+
add_3_readvariableop_resource:=3
 matmul_4_readvariableop_resource:	=?,
add_4_readvariableop_resource:	?3
 matmul_5_readvariableop_resource:	?+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?add/ReadVariableOp?add_1/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:t*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????tZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????v
dropout/dropout/MulMulTanh:y:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????tM
dropout/dropout/ShapeShapeTanh:y:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????t*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????t
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????t?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????tx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ty*
dtype0?
MatMul_1MatMuldropout/dropout/Mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:y*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????y\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????|
dropout_1/dropout/MulMul
Tanh_1:y:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????yQ
dropout_1/dropout/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????y*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????y?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????y?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????yy
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	y?*
dtype0?
MatMul_2MatMuldropout_1/dropout/Mul_1:z:0MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????P
SigmoidSigmoid	add_2:z:0*
T0*(
_output_shapes
:??????????\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????~
dropout_2/dropout/MulMulSigmoid:y:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????R
dropout_2/dropout/ShapeShapeSigmoid:y:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????y
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?=*
dtype0?
MatMul_3MatMuldropout_2/dropout/Mul_1:z:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:=*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=Q
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:?????????=\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????
dropout_3/dropout/MulMulSigmoid_1:y:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????=T
dropout_3/dropout/ShapeShapeSigmoid_1:y:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????=*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????=?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????=?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????=y
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes
:	=?*
dtype0?
MatMul_4MatMuldropout_3/dropout/Mul_1:z:0MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????K
	LeakyRelu	LeakyRelu	add_4:z:0*(
_output_shapes
:??????????\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_4/dropout/MulMulLeakyRelu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:??????????^
dropout_4/dropout/ShapeShapeLeakyRelu:activations:0*
T0*
_output_shapes
:?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????y
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMul_5MatMuldropout_4/dropout/Mul_1:z:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:?????????x
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp:J F
'
_output_shapes
:?????????

_user_specified_nameX
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_37292

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????t[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????t"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????t:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_37331

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????yC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????y*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????yo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????yi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????yY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????y"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????y:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_36748

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????tC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????t*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????to
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????ti
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????tY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????t"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????t:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?I
?
 __inference__wrapped_model_36514
input_1@
.sobolev_network_matmul_readvariableop_resource:t9
+sobolev_network_add_readvariableop_resource:tB
0sobolev_network_matmul_1_readvariableop_resource:ty;
-sobolev_network_add_1_readvariableop_resource:yC
0sobolev_network_matmul_2_readvariableop_resource:	y?<
-sobolev_network_add_2_readvariableop_resource:	?C
0sobolev_network_matmul_3_readvariableop_resource:	?=;
-sobolev_network_add_3_readvariableop_resource:=C
0sobolev_network_matmul_4_readvariableop_resource:	=?<
-sobolev_network_add_4_readvariableop_resource:	?C
0sobolev_network_matmul_5_readvariableop_resource:	?;
-sobolev_network_add_5_readvariableop_resource:B
0sobolev_network_matmul_6_readvariableop_resource:;
-sobolev_network_add_6_readvariableop_resource:
identity??%sobolev_network/MatMul/ReadVariableOp?'sobolev_network/MatMul_1/ReadVariableOp?'sobolev_network/MatMul_2/ReadVariableOp?'sobolev_network/MatMul_3/ReadVariableOp?'sobolev_network/MatMul_4/ReadVariableOp?'sobolev_network/MatMul_5/ReadVariableOp?'sobolev_network/MatMul_6/ReadVariableOp?"sobolev_network/add/ReadVariableOp?$sobolev_network/add_1/ReadVariableOp?$sobolev_network/add_2/ReadVariableOp?$sobolev_network/add_3/ReadVariableOp?$sobolev_network/add_4/ReadVariableOp?$sobolev_network/add_5/ReadVariableOp?$sobolev_network/add_6/ReadVariableOp?
%sobolev_network/MatMul/ReadVariableOpReadVariableOp.sobolev_network_matmul_readvariableop_resource*
_output_shapes

:t*
dtype0?
sobolev_network/MatMulMatMulinput_1-sobolev_network/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t?
"sobolev_network/add/ReadVariableOpReadVariableOp+sobolev_network_add_readvariableop_resource*
_output_shapes
:t*
dtype0?
sobolev_network/addAddV2 sobolev_network/MatMul:product:0*sobolev_network/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tg
sobolev_network/TanhTanhsobolev_network/add:z:0*
T0*'
_output_shapes
:?????????tx
 sobolev_network/dropout/IdentityIdentitysobolev_network/Tanh:y:0*
T0*'
_output_shapes
:?????????t?
'sobolev_network/MatMul_1/ReadVariableOpReadVariableOp0sobolev_network_matmul_1_readvariableop_resource*
_output_shapes

:ty*
dtype0?
sobolev_network/MatMul_1MatMul)sobolev_network/dropout/Identity:output:0/sobolev_network/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y?
$sobolev_network/add_1/ReadVariableOpReadVariableOp-sobolev_network_add_1_readvariableop_resource*
_output_shapes
:y*
dtype0?
sobolev_network/add_1AddV2"sobolev_network/MatMul_1:product:0,sobolev_network/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yk
sobolev_network/Tanh_1Tanhsobolev_network/add_1:z:0*
T0*'
_output_shapes
:?????????y|
"sobolev_network/dropout_1/IdentityIdentitysobolev_network/Tanh_1:y:0*
T0*'
_output_shapes
:?????????y?
'sobolev_network/MatMul_2/ReadVariableOpReadVariableOp0sobolev_network_matmul_2_readvariableop_resource*
_output_shapes
:	y?*
dtype0?
sobolev_network/MatMul_2MatMul+sobolev_network/dropout_1/Identity:output:0/sobolev_network/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$sobolev_network/add_2/ReadVariableOpReadVariableOp-sobolev_network_add_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sobolev_network/add_2AddV2"sobolev_network/MatMul_2:product:0,sobolev_network/add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????p
sobolev_network/SigmoidSigmoidsobolev_network/add_2:z:0*
T0*(
_output_shapes
:??????????~
"sobolev_network/dropout_2/IdentityIdentitysobolev_network/Sigmoid:y:0*
T0*(
_output_shapes
:???????????
'sobolev_network/MatMul_3/ReadVariableOpReadVariableOp0sobolev_network_matmul_3_readvariableop_resource*
_output_shapes
:	?=*
dtype0?
sobolev_network/MatMul_3MatMul+sobolev_network/dropout_2/Identity:output:0/sobolev_network/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=?
$sobolev_network/add_3/ReadVariableOpReadVariableOp-sobolev_network_add_3_readvariableop_resource*
_output_shapes
:=*
dtype0?
sobolev_network/add_3AddV2"sobolev_network/MatMul_3:product:0,sobolev_network/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=q
sobolev_network/Sigmoid_1Sigmoidsobolev_network/add_3:z:0*
T0*'
_output_shapes
:?????????=
"sobolev_network/dropout_3/IdentityIdentitysobolev_network/Sigmoid_1:y:0*
T0*'
_output_shapes
:?????????=?
'sobolev_network/MatMul_4/ReadVariableOpReadVariableOp0sobolev_network_matmul_4_readvariableop_resource*
_output_shapes
:	=?*
dtype0?
sobolev_network/MatMul_4MatMul+sobolev_network/dropout_3/Identity:output:0/sobolev_network/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$sobolev_network/add_4/ReadVariableOpReadVariableOp-sobolev_network_add_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sobolev_network/add_4AddV2"sobolev_network/MatMul_4:product:0,sobolev_network/add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
sobolev_network/LeakyRelu	LeakyRelusobolev_network/add_4:z:0*(
_output_shapes
:???????????
"sobolev_network/dropout_4/IdentityIdentity'sobolev_network/LeakyRelu:activations:0*
T0*(
_output_shapes
:???????????
'sobolev_network/MatMul_5/ReadVariableOpReadVariableOp0sobolev_network_matmul_5_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sobolev_network/MatMul_5MatMul+sobolev_network/dropout_4/Identity:output:0/sobolev_network/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$sobolev_network/add_5/ReadVariableOpReadVariableOp-sobolev_network_add_5_readvariableop_resource*
_output_shapes
:*
dtype0?
sobolev_network/add_5AddV2"sobolev_network/MatMul_5:product:0,sobolev_network/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????i
sobolev_network/ReluRelusobolev_network/add_5:z:0*
T0*'
_output_shapes
:??????????
'sobolev_network/MatMul_6/ReadVariableOpReadVariableOp0sobolev_network_matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0?
sobolev_network/MatMul_6MatMul"sobolev_network/Relu:activations:0/sobolev_network/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$sobolev_network/add_6/ReadVariableOpReadVariableOp-sobolev_network_add_6_readvariableop_resource*
_output_shapes
:*
dtype0?
sobolev_network/add_6AddV2"sobolev_network/MatMul_6:product:0,sobolev_network/add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitysobolev_network/add_6:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sobolev_network/MatMul/ReadVariableOp(^sobolev_network/MatMul_1/ReadVariableOp(^sobolev_network/MatMul_2/ReadVariableOp(^sobolev_network/MatMul_3/ReadVariableOp(^sobolev_network/MatMul_4/ReadVariableOp(^sobolev_network/MatMul_5/ReadVariableOp(^sobolev_network/MatMul_6/ReadVariableOp#^sobolev_network/add/ReadVariableOp%^sobolev_network/add_1/ReadVariableOp%^sobolev_network/add_2/ReadVariableOp%^sobolev_network/add_3/ReadVariableOp%^sobolev_network/add_4/ReadVariableOp%^sobolev_network/add_5/ReadVariableOp%^sobolev_network/add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2N
%sobolev_network/MatMul/ReadVariableOp%sobolev_network/MatMul/ReadVariableOp2R
'sobolev_network/MatMul_1/ReadVariableOp'sobolev_network/MatMul_1/ReadVariableOp2R
'sobolev_network/MatMul_2/ReadVariableOp'sobolev_network/MatMul_2/ReadVariableOp2R
'sobolev_network/MatMul_3/ReadVariableOp'sobolev_network/MatMul_3/ReadVariableOp2R
'sobolev_network/MatMul_4/ReadVariableOp'sobolev_network/MatMul_4/ReadVariableOp2R
'sobolev_network/MatMul_5/ReadVariableOp'sobolev_network/MatMul_5/ReadVariableOp2R
'sobolev_network/MatMul_6/ReadVariableOp'sobolev_network/MatMul_6/ReadVariableOp2H
"sobolev_network/add/ReadVariableOp"sobolev_network/add/ReadVariableOp2L
$sobolev_network/add_1/ReadVariableOp$sobolev_network/add_1/ReadVariableOp2L
$sobolev_network/add_2/ReadVariableOp$sobolev_network/add_2/ReadVariableOp2L
$sobolev_network/add_3/ReadVariableOp$sobolev_network/add_3/ReadVariableOp2L
$sobolev_network/add_4/ReadVariableOp$sobolev_network/add_4/ReadVariableOp2L
$sobolev_network/add_5/ReadVariableOp$sobolev_network/add_5/ReadVariableOp2L
$sobolev_network/add_6/ReadVariableOp$sobolev_network/add_6/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?>
?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36964
input_10
matmul_readvariableop_resource:t)
add_readvariableop_resource:t2
 matmul_1_readvariableop_resource:ty+
add_1_readvariableop_resource:y3
 matmul_2_readvariableop_resource:	y?,
add_2_readvariableop_resource:	?3
 matmul_3_readvariableop_resource:	?=+
add_3_readvariableop_resource:=3
 matmul_4_readvariableop_resource:	=?,
add_4_readvariableop_resource:	?3
 matmul_5_readvariableop_resource:	?+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?add/ReadVariableOp?add_1/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t*
dtype0j
MatMulMatMulinput_1MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:t*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????tG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????t?
dropout/PartitionedCallPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_36533x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ty*
dtype0?
MatMul_1MatMul dropout/PartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:y*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????yK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????y?
dropout_1/PartitionedCallPartitionedCall
Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_36547y
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes
:	y?*
dtype0?
MatMul_2MatMul"dropout_1/PartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????P
SigmoidSigmoid	add_2:z:0*
T0*(
_output_shapes
:???????????
dropout_2/PartitionedCallPartitionedCallSigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_36561y
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes
:	?=*
dtype0?
MatMul_3MatMul"dropout_2/PartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:=*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????=Q
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:?????????=?
dropout_3/PartitionedCallPartitionedCallSigmoid_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_36575y
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes
:	=?*
dtype0?
MatMul_4MatMul"dropout_3/PartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes	
:?*
dtype0s
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????K
	LeakyRelu	LeakyRelu	add_4:z:0*(
_output_shapes
:???????????
dropout_4/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_36589y
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes
:	?*
dtype0?
MatMul_5MatMul"dropout_4/PartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????I
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:?????????x
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
??
? 
!__inference__traced_restore_37787
file_prefix.
assignvariableop_variable_13:t,
assignvariableop_1_variable_12:t0
assignvariableop_2_variable_11:ty,
assignvariableop_3_variable_10:y0
assignvariableop_4_variable_9:	y?,
assignvariableop_5_variable_8:	?0
assignvariableop_6_variable_7:	?=+
assignvariableop_7_variable_6:=0
assignvariableop_8_variable_5:	=?,
assignvariableop_9_variable_4:	?1
assignvariableop_10_variable_3:	?,
assignvariableop_11_variable_2:0
assignvariableop_12_variable_1:*
assignvariableop_13_variable:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: %
assignvariableop_19_total_4: %
assignvariableop_20_count_4: %
assignvariableop_21_total_3: %
assignvariableop_22_count_3: %
assignvariableop_23_total_2: %
assignvariableop_24_count_2: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: 8
&assignvariableop_29_adam_variable_m_13:t4
&assignvariableop_30_adam_variable_m_12:t8
&assignvariableop_31_adam_variable_m_11:ty4
&assignvariableop_32_adam_variable_m_10:y8
%assignvariableop_33_adam_variable_m_9:	y?4
%assignvariableop_34_adam_variable_m_8:	?8
%assignvariableop_35_adam_variable_m_7:	?=3
%assignvariableop_36_adam_variable_m_6:=8
%assignvariableop_37_adam_variable_m_5:	=?4
%assignvariableop_38_adam_variable_m_4:	?8
%assignvariableop_39_adam_variable_m_3:	?3
%assignvariableop_40_adam_variable_m_2:7
%assignvariableop_41_adam_variable_m_1:1
#assignvariableop_42_adam_variable_m:8
&assignvariableop_43_adam_variable_v_13:t4
&assignvariableop_44_adam_variable_v_12:t8
&assignvariableop_45_adam_variable_v_11:ty4
&assignvariableop_46_adam_variable_v_10:y8
%assignvariableop_47_adam_variable_v_9:	y?4
%assignvariableop_48_adam_variable_v_8:	?8
%assignvariableop_49_adam_variable_v_7:	?=3
%assignvariableop_50_adam_variable_v_6:=8
%assignvariableop_51_adam_variable_v_5:	=?4
%assignvariableop_52_adam_variable_v_4:	?8
%assignvariableop_53_adam_variable_v_3:	?3
%assignvariableop_54_adam_variable_v_2:7
%assignvariableop_55_adam_variable_v_1:1
#assignvariableop_56_adam_variable_v:
identity_58??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:BW1/.ATTRIBUTES/VARIABLE_VALUEBb1/.ATTRIBUTES/VARIABLE_VALUEBW2/.ATTRIBUTES/VARIABLE_VALUEBb2/.ATTRIBUTES/VARIABLE_VALUEBW3/.ATTRIBUTES/VARIABLE_VALUEBb3/.ATTRIBUTES/VARIABLE_VALUEBW4/.ATTRIBUTES/VARIABLE_VALUEBb4/.ATTRIBUTES/VARIABLE_VALUEBW5/.ATTRIBUTES/VARIABLE_VALUEBb5/.ATTRIBUTES/VARIABLE_VALUEBW6/.ATTRIBUTES/VARIABLE_VALUEBb6/.ATTRIBUTES/VARIABLE_VALUEBW7/.ATTRIBUTES/VARIABLE_VALUEBb7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_variable_13Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_12Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_11Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_10Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_9Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_8Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_7Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_6Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_5Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_4Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_3Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_variableIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_4Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_4Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_3Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_3Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_variable_m_13Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_variable_m_12Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_adam_variable_m_11Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_variable_m_10Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_variable_m_9Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_variable_m_8Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp%assignvariableop_35_adam_variable_m_7Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_variable_m_6Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_variable_m_5Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_variable_m_4Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_variable_m_3Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_variable_m_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_variable_m_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp#assignvariableop_42_adam_variable_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp&assignvariableop_43_adam_variable_v_13Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_variable_v_12Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_variable_v_11Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_variable_v_10Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp%assignvariableop_47_adam_variable_v_9Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_variable_v_8Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp%assignvariableop_49_adam_variable_v_7Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp%assignvariableop_50_adam_variable_v_6Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp%assignvariableop_51_adam_variable_v_5Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_variable_v_4Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp%assignvariableop_53_adam_variable_v_3Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp%assignvariableop_54_adam_variable_v_2Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp%assignvariableop_55_adam_variable_v_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp#assignvariableop_56_adam_variable_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: ?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*?
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_37358

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
W1
	b1

dp1
W2
b2
dp2
W3
b3
dp3
W4
b4
dp4
W5
b5
dp5
W6
b6
dp6
W7
b7
w
	optimizer

signatures"
_tf_keras_model
?
0
	1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
?
0
	1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
$trace_0
%trace_1
&trace_2
'trace_32?
/__inference_sobolev_network_layer_call_fn_36636
/__inference_sobolev_network_layer_call_fn_37095
/__inference_sobolev_network_layer_call_fn_37128
/__inference_sobolev_network_layer_call_fn_36907?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z$trace_0z%trace_1z&trace_2z'trace_3
?
(trace_0
)trace_1
*trace_2
+trace_32?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37185
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37277
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36964
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37021?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z(trace_0z)trace_1z*trace_2z+trace_3
?B?
 __inference__wrapped_model_36514input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:t2Variable
:t2Variable
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator"
_tf_keras_layer
:ty2Variable
:y2Variable
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator"
_tf_keras_layer
:	y?2Variable
:?2Variable
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@_random_generator"
_tf_keras_layer
:	?=2Variable
:=2Variable
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator"
_tf_keras_layer
:	=?2Variable
:?2Variable
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_random_generator"
_tf_keras_layer
:	?2Variable
:2Variable
?
O	keras_api
P_random_generator"
_tf_keras_layer
:2Variable
:2Variable
Q
Q0
R1
S2
T3
U4
V5
W6"
trackable_list_wrapper
?
Xiter

Ybeta_1

Zbeta_2
	[decay
\learning_ratem?	m?m?m?m?m?m?m?m?m?m?m?m?m?v?	v?v?v?v?v?v?v?v?v?v?v?v?v?"
	optimizer
,
]serving_default"
signature_map
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
C
^0
_1
`2
a3
b4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_sobolev_network_layer_call_fn_36636input_1"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
/__inference_sobolev_network_layer_call_fn_37095X"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
/__inference_sobolev_network_layer_call_fn_37128X"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
/__inference_sobolev_network_layer_call_fn_36907input_1"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37185X"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37277X"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36964input_1"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37021input_1"?
???
FullArgSpec
args?
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?
htrace_0
itrace_12?
'__inference_dropout_layer_call_fn_37282
'__inference_dropout_layer_call_fn_37287?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zhtrace_0zitrace_1
?
jtrace_0
ktrace_12?
B__inference_dropout_layer_call_and_return_conditional_losses_37292
B__inference_dropout_layer_call_and_return_conditional_losses_37304?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zjtrace_0zktrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
?
qtrace_0
rtrace_12?
)__inference_dropout_1_layer_call_fn_37309
)__inference_dropout_1_layer_call_fn_37314?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zqtrace_0zrtrace_1
?
strace_0
ttrace_12?
D__inference_dropout_1_layer_call_and_return_conditional_losses_37319
D__inference_dropout_1_layer_call_and_return_conditional_losses_37331?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zstrace_0zttrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?
ztrace_0
{trace_12?
)__inference_dropout_2_layer_call_fn_37336
)__inference_dropout_2_layer_call_fn_37341?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zztrace_0z{trace_1
?
|trace_0
}trace_12?
D__inference_dropout_2_layer_call_and_return_conditional_losses_37346
D__inference_dropout_2_layer_call_and_return_conditional_losses_37358?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z|trace_0z}trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables

layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_3_layer_call_fn_37363
)__inference_dropout_3_layer_call_fn_37368?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_3_layer_call_and_return_conditional_losses_37373
D__inference_dropout_3_layer_call_and_return_conditional_losses_37385?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_4_layer_call_fn_37390
)__inference_dropout_4_layer_call_fn_37395?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_4_layer_call_and_return_conditional_losses_37400
D__inference_dropout_4_layer_call_and_return_conditional_losses_37412?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
/
0
	1"
trackable_tuple_wrapper
/
0
1"
trackable_tuple_wrapper
/
0
1"
trackable_tuple_wrapper
/
0
1"
trackable_tuple_wrapper
/
0
1"
trackable_tuple_wrapper
/
0
1"
trackable_tuple_wrapper
/
0
1"
trackable_tuple_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_37062input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dropout_layer_call_fn_37282inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_dropout_layer_call_fn_37287inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_37292inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_37304inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_dropout_1_layer_call_fn_37309inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
)__inference_dropout_1_layer_call_fn_37314inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_37319inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_37331inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_dropout_2_layer_call_fn_37336inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
)__inference_dropout_2_layer_call_fn_37341inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_37346inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_37358inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_dropout_3_layer_call_fn_37363inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
)__inference_dropout_3_layer_call_fn_37368inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_3_layer_call_and_return_conditional_losses_37373inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_3_layer_call_and_return_conditional_losses_37385inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_dropout_4_layer_call_fn_37390inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
)__inference_dropout_4_layer_call_fn_37395inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_4_layer_call_and_return_conditional_losses_37400inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dropout_4_layer_call_and_return_conditional_losses_37412inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
:t2Adam/Variable/m
:t2Adam/Variable/m
:ty2Adam/Variable/m
:y2Adam/Variable/m
 :	y?2Adam/Variable/m
:?2Adam/Variable/m
 :	?=2Adam/Variable/m
:=2Adam/Variable/m
 :	=?2Adam/Variable/m
:?2Adam/Variable/m
 :	?2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:t2Adam/Variable/v
:t2Adam/Variable/v
:ty2Adam/Variable/v
:y2Adam/Variable/v
 :	y?2Adam/Variable/v
:?2Adam/Variable/v
 :	?=2Adam/Variable/v
:=2Adam/Variable/v
 :	=?2Adam/Variable/v
:?2Adam/Variable/v
 :	?2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v?
 __inference__wrapped_model_36514w	0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_37319\3?0
)?&
 ?
inputs?????????y
p 
? "%?"
?
0?????????y
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_37331\3?0
)?&
 ?
inputs?????????y
p
? "%?"
?
0?????????y
? |
)__inference_dropout_1_layer_call_fn_37309O3?0
)?&
 ?
inputs?????????y
p 
? "??????????y|
)__inference_dropout_1_layer_call_fn_37314O3?0
)?&
 ?
inputs?????????y
p
? "??????????y?
D__inference_dropout_2_layer_call_and_return_conditional_losses_37346^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_37358^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ~
)__inference_dropout_2_layer_call_fn_37336Q4?1
*?'
!?
inputs??????????
p 
? "???????????~
)__inference_dropout_2_layer_call_fn_37341Q4?1
*?'
!?
inputs??????????
p
? "????????????
D__inference_dropout_3_layer_call_and_return_conditional_losses_37373\3?0
)?&
 ?
inputs?????????=
p 
? "%?"
?
0?????????=
? ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_37385\3?0
)?&
 ?
inputs?????????=
p
? "%?"
?
0?????????=
? |
)__inference_dropout_3_layer_call_fn_37363O3?0
)?&
 ?
inputs?????????=
p 
? "??????????=|
)__inference_dropout_3_layer_call_fn_37368O3?0
)?&
 ?
inputs?????????=
p
? "??????????=?
D__inference_dropout_4_layer_call_and_return_conditional_losses_37400^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_37412^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ~
)__inference_dropout_4_layer_call_fn_37390Q4?1
*?'
!?
inputs??????????
p 
? "???????????~
)__inference_dropout_4_layer_call_fn_37395Q4?1
*?'
!?
inputs??????????
p
? "????????????
B__inference_dropout_layer_call_and_return_conditional_losses_37292\3?0
)?&
 ?
inputs?????????t
p 
? "%?"
?
0?????????t
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_37304\3?0
)?&
 ?
inputs?????????t
p
? "%?"
?
0?????????t
? z
'__inference_dropout_layer_call_fn_37282O3?0
)?&
 ?
inputs?????????t
p 
? "??????????tz
'__inference_dropout_layer_call_fn_37287O3?0
)?&
 ?
inputs?????????t
p
? "??????????t?
#__inference_signature_wrapper_37062?	;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1??????????
J__inference_sobolev_network_layer_call_and_return_conditional_losses_36964y	@?=
&?#
!?
input_1?????????
?

trainingp "%?"
?
0?????????
? ?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37021y	@?=
&?#
!?
input_1?????????
?

trainingp"%?"
?
0?????????
? ?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37185s	:?7
 ?
?
X?????????
?

trainingp "%?"
?
0?????????
? ?
J__inference_sobolev_network_layer_call_and_return_conditional_losses_37277s	:?7
 ?
?
X?????????
?

trainingp"%?"
?
0?????????
? ?
/__inference_sobolev_network_layer_call_fn_36636l	@?=
&?#
!?
input_1?????????
?

trainingp "???????????
/__inference_sobolev_network_layer_call_fn_36907l	@?=
&?#
!?
input_1?????????
?

trainingp"???????????
/__inference_sobolev_network_layer_call_fn_37095f	:?7
 ?
?
X?????????
?

trainingp "???????????
/__inference_sobolev_network_layer_call_fn_37128f	:?7
 ?
?
X?????????
?

trainingp"??????????