¥þ

ùÉ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
alphafloat%ÍÌL>"
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8¬	
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
:*"
shared_nameAdam/Variable/v_1
w
%Adam/Variable/v_1/Read/ReadVariableOpReadVariableOpAdam/Variable/v_1*
_output_shapes

:*
dtype0
z
Adam/Variable/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/v_2
s
%Adam/Variable/v_2/Read/ReadVariableOpReadVariableOpAdam/Variable/v_2*
_output_shapes
:*
dtype0
~
Adam/Variable/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/v_3
w
%Adam/Variable/v_3/Read/ReadVariableOpReadVariableOpAdam/Variable/v_3*
_output_shapes

:*
dtype0
z
Adam/Variable/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/v_4
s
%Adam/Variable/v_4/Read/ReadVariableOpReadVariableOpAdam/Variable/v_4*
_output_shapes
:*
dtype0
~
Adam/Variable/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/v_5
w
%Adam/Variable/v_5/Read/ReadVariableOpReadVariableOpAdam/Variable/v_5*
_output_shapes

:*
dtype0
z
Adam/Variable/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/v_6
s
%Adam/Variable/v_6/Read/ReadVariableOpReadVariableOpAdam/Variable/v_6*
_output_shapes
:*
dtype0
~
Adam/Variable/v_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/v_7
w
%Adam/Variable/v_7/Read/ReadVariableOpReadVariableOpAdam/Variable/v_7*
_output_shapes

:*
dtype0
z
Adam/Variable/v_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/v_8
s
%Adam/Variable/v_8/Read/ReadVariableOpReadVariableOpAdam/Variable/v_8*
_output_shapes
:*
dtype0
~
Adam/Variable/v_9VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/v_9
w
%Adam/Variable/v_9/Read/ReadVariableOpReadVariableOpAdam/Variable/v_9*
_output_shapes

:*
dtype0
|
Adam/Variable/v_10VarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Variable/v_10
u
&Adam/Variable/v_10/Read/ReadVariableOpReadVariableOpAdam/Variable/v_10*
_output_shapes
:*
dtype0

Adam/Variable/v_11VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameAdam/Variable/v_11
y
&Adam/Variable/v_11/Read/ReadVariableOpReadVariableOpAdam/Variable/v_11*
_output_shapes

:*
dtype0
|
Adam/Variable/v_12VarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Variable/v_12
u
&Adam/Variable/v_12/Read/ReadVariableOpReadVariableOpAdam/Variable/v_12*
_output_shapes
:*
dtype0

Adam/Variable/v_13VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameAdam/Variable/v_13
y
&Adam/Variable/v_13/Read/ReadVariableOpReadVariableOpAdam/Variable/v_13*
_output_shapes

:*
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
:*"
shared_nameAdam/Variable/m_1
w
%Adam/Variable/m_1/Read/ReadVariableOpReadVariableOpAdam/Variable/m_1*
_output_shapes

:*
dtype0
z
Adam/Variable/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/m_2
s
%Adam/Variable/m_2/Read/ReadVariableOpReadVariableOpAdam/Variable/m_2*
_output_shapes
:*
dtype0
~
Adam/Variable/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/m_3
w
%Adam/Variable/m_3/Read/ReadVariableOpReadVariableOpAdam/Variable/m_3*
_output_shapes

:*
dtype0
z
Adam/Variable/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/m_4
s
%Adam/Variable/m_4/Read/ReadVariableOpReadVariableOpAdam/Variable/m_4*
_output_shapes
:*
dtype0
~
Adam/Variable/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/m_5
w
%Adam/Variable/m_5/Read/ReadVariableOpReadVariableOpAdam/Variable/m_5*
_output_shapes

:*
dtype0
z
Adam/Variable/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/m_6
s
%Adam/Variable/m_6/Read/ReadVariableOpReadVariableOpAdam/Variable/m_6*
_output_shapes
:*
dtype0
~
Adam/Variable/m_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/m_7
w
%Adam/Variable/m_7/Read/ReadVariableOpReadVariableOpAdam/Variable/m_7*
_output_shapes

:*
dtype0
z
Adam/Variable/m_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Variable/m_8
s
%Adam/Variable/m_8/Read/ReadVariableOpReadVariableOpAdam/Variable/m_8*
_output_shapes
:*
dtype0
~
Adam/Variable/m_9VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/Variable/m_9
w
%Adam/Variable/m_9/Read/ReadVariableOpReadVariableOpAdam/Variable/m_9*
_output_shapes

:*
dtype0
|
Adam/Variable/m_10VarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Variable/m_10
u
&Adam/Variable/m_10/Read/ReadVariableOpReadVariableOpAdam/Variable/m_10*
_output_shapes
:*
dtype0

Adam/Variable/m_11VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameAdam/Variable/m_11
y
&Adam/Variable/m_11/Read/ReadVariableOpReadVariableOpAdam/Variable/m_11*
_output_shapes

:*
dtype0
|
Adam/Variable/m_12VarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Variable/m_12
u
&Adam/Variable/m_12/Read/ReadVariableOpReadVariableOpAdam/Variable/m_12*
_output_shapes
:*
dtype0

Adam/Variable/m_13VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameAdam/Variable/m_13
y
&Adam/Variable/m_13/Read/ReadVariableOpReadVariableOpAdam/Variable/m_13*
_output_shapes

:*
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
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
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
:*
shared_name
Variable_1
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:*
dtype0
l

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_2
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0
p

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable_3
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes

:*
dtype0
l

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_4
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:*
dtype0
p

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable_5
i
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes

:*
dtype0
l

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_6
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:*
dtype0
p

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable_7
i
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes

:*
dtype0
l

Variable_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Variable_8
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:*
dtype0
p

Variable_9VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable_9
i
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes

:*
dtype0
n
Variable_10VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameVariable_10
g
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:*
dtype0
r
Variable_11VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameVariable_11
k
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes

:*
dtype0
n
Variable_12VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameVariable_12
g
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:*
dtype0
r
Variable_13VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameVariable_13
k
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes

:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ö
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_7689

NoOpNoOp
G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÍF
valueÃFBÀF B¹F
ù
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
°
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
¥
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
¥
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
¥
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
¥
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
¥
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
Ü

Xbeta_1

Ybeta_2
	Zdecay
[learning_rate
\iterm	mmmmmmmmmmmmmv	vv v¡v¢v£v¤v¥v¦v§v¨v©vªv«*
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

^0*
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

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 
* 
* 
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

mtrace_0
ntrace_1* 

otrace_0
ptrace_1* 
* 
* 
* 
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

vtrace_0
wtrace_1* 

xtrace_0
ytrace_1* 
* 
* 
* 
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
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
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
	variables
	keras_api

total

count*
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
0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_13/Read/ReadVariableOpVariable_12/Read/ReadVariableOpVariable_11/Read/ReadVariableOpVariable_10/Read/ReadVariableOpVariable_9/Read/ReadVariableOpVariable_8/Read/ReadVariableOpVariable_7/Read/ReadVariableOpVariable_6/Read/ReadVariableOpVariable_5/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp&Adam/Variable/m_13/Read/ReadVariableOp&Adam/Variable/m_12/Read/ReadVariableOp&Adam/Variable/m_11/Read/ReadVariableOp&Adam/Variable/m_10/Read/ReadVariableOp%Adam/Variable/m_9/Read/ReadVariableOp%Adam/Variable/m_8/Read/ReadVariableOp%Adam/Variable/m_7/Read/ReadVariableOp%Adam/Variable/m_6/Read/ReadVariableOp%Adam/Variable/m_5/Read/ReadVariableOp%Adam/Variable/m_4/Read/ReadVariableOp%Adam/Variable/m_3/Read/ReadVariableOp%Adam/Variable/m_2/Read/ReadVariableOp%Adam/Variable/m_1/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp&Adam/Variable/v_13/Read/ReadVariableOp&Adam/Variable/v_12/Read/ReadVariableOp&Adam/Variable/v_11/Read/ReadVariableOp&Adam/Variable/v_10/Read/ReadVariableOp%Adam/Variable/v_9/Read/ReadVariableOp%Adam/Variable/v_8/Read/ReadVariableOp%Adam/Variable/v_7/Read/ReadVariableOp%Adam/Variable/v_6/Read/ReadVariableOp%Adam/Variable/v_5/Read/ReadVariableOp%Adam/Variable/v_4/Read/ReadVariableOp%Adam/Variable/v_3/Read/ReadVariableOp%Adam/Variable/v_2/Read/ReadVariableOp%Adam/Variable/v_1/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOpConst*>
Tin7
523	*
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
GPU 2J 8 *&
f!R
__inference__traced_save_8209
Æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablebeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/Variable/m_13Adam/Variable/m_12Adam/Variable/m_11Adam/Variable/m_10Adam/Variable/m_9Adam/Variable/m_8Adam/Variable/m_7Adam/Variable/m_6Adam/Variable/m_5Adam/Variable/m_4Adam/Variable/m_3Adam/Variable/m_2Adam/Variable/m_1Adam/Variable/mAdam/Variable/v_13Adam/Variable/v_12Adam/Variable/v_11Adam/Variable/v_10Adam/Variable/v_9Adam/Variable/v_8Adam/Variable/v_7Adam/Variable/v_6Adam/Variable/v_5Adam/Variable/v_4Adam/Variable/v_3Adam/Variable/v_2Adam/Variable/v_1Adam/Variable/v*=
Tin6
422*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_8366¥î

×
"__inference_signature_wrapper_7689
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallÔ
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_7141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
îY
Ù
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7904
x0
matmul_readvariableop_resource:)
add_readvariableop_resource:2
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:+
add_2_readvariableop_resource:2
 matmul_3_readvariableop_resource:+
add_3_readvariableop_resource:2
 matmul_4_readvariableop_resource:+
add_4_readvariableop_resource:2
 matmul_5_readvariableop_resource:+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢MatMul_3/ReadVariableOp¢MatMul_4/ReadVariableOp¢MatMul_5/ReadVariableOp¢MatMul_6/ReadVariableOp¢add/ReadVariableOp¢add_1/ReadVariableOp¢add_2/ReadVariableOp¢add_3/ReadVariableOp¢add_4/ReadVariableOp¢add_5/ReadVariableOp¢add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dropout/dropout/MulMulTanh:y:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
dropout/dropout/ShapeShapeTanh:y:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¾
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_1MatMuldropout/dropout/Mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?|
dropout_1/dropout/MulMul
Tanh_1:y:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dropout_1/dropout/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
: 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_2MatMuldropout_1/dropout/Mul_1:z:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
SigmoidSigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?}
dropout_2/dropout/MulMulSigmoid:y:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout_2/dropout/ShapeShapeSigmoid:y:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_3MatMuldropout_2/dropout/Mul_1:z:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_3/dropout/MulMulSigmoid_1:y:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_3/dropout/ShapeShapeSigmoid_1:y:0*
T0*
_output_shapes
: 
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_4MatMuldropout_3/dropout/Mul_1:z:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype0r
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
	LeakyRelu	LeakyRelu	add_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_4/dropout/MulMulLeakyRelu:activations:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dropout_4/dropout/ShapeShapeLeakyRelu:activations:0*
T0*
_output_shapes
: 
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_5MatMuldropout_4/dropout/Mul_1:z:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2.
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameX
í
a
(__inference_dropout_3_layer_call_fn_7995

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7306o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
ã
.__inference_sobolev_network_layer_call_fn_7263
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallþ
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ñ	
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7352

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÒW
Ú
__inference__traced_save_8209
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
#savev2_variable_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
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

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ª
value B2BW1/.ATTRIBUTES/VARIABLE_VALUEBb1/.ATTRIBUTES/VARIABLE_VALUEBW2/.ATTRIBUTES/VARIABLE_VALUEBb2/.ATTRIBUTES/VARIABLE_VALUEBW3/.ATTRIBUTES/VARIABLE_VALUEBb3/.ATTRIBUTES/VARIABLE_VALUEBW4/.ATTRIBUTES/VARIABLE_VALUEBb4/.ATTRIBUTES/VARIABLE_VALUEBW5/.ATTRIBUTES/VARIABLE_VALUEBb5/.ATTRIBUTES/VARIABLE_VALUEBW6/.ATTRIBUTES/VARIABLE_VALUEBb6/.ATTRIBUTES/VARIABLE_VALUEBW7/.ATTRIBUTES/VARIABLE_VALUEBb7/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÑ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_variable_13_read_readvariableop&savev2_variable_12_read_readvariableop&savev2_variable_11_read_readvariableop&savev2_variable_10_read_readvariableop%savev2_variable_9_read_readvariableop%savev2_variable_8_read_readvariableop%savev2_variable_7_read_readvariableop%savev2_variable_6_read_readvariableop%savev2_variable_5_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_1_read_readvariableop#savev2_variable_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop-savev2_adam_variable_m_13_read_readvariableop-savev2_adam_variable_m_12_read_readvariableop-savev2_adam_variable_m_11_read_readvariableop-savev2_adam_variable_m_10_read_readvariableop,savev2_adam_variable_m_9_read_readvariableop,savev2_adam_variable_m_8_read_readvariableop,savev2_adam_variable_m_7_read_readvariableop,savev2_adam_variable_m_6_read_readvariableop,savev2_adam_variable_m_5_read_readvariableop,savev2_adam_variable_m_4_read_readvariableop,savev2_adam_variable_m_3_read_readvariableop,savev2_adam_variable_m_2_read_readvariableop,savev2_adam_variable_m_1_read_readvariableop*savev2_adam_variable_m_read_readvariableop-savev2_adam_variable_v_13_read_readvariableop-savev2_adam_variable_v_12_read_readvariableop-savev2_adam_variable_v_11_read_readvariableop-savev2_adam_variable_v_10_read_readvariableop,savev2_adam_variable_v_9_read_readvariableop,savev2_adam_variable_v_8_read_readvariableop,savev2_adam_variable_v_7_read_readvariableop,savev2_adam_variable_v_6_read_readvariableop,savev2_adam_variable_v_5_read_readvariableop,savev2_adam_variable_v_4_read_readvariableop,savev2_adam_variable_v_3_read_readvariableop,savev2_adam_variable_v_2_read_readvariableop,savev2_adam_variable_v_1_read_readvariableop*savev2_adam_variable_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*÷
_input_shapeså
â: ::::::::::::::: : : : : : : ::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 
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
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::2

_output_shapes
: 

D
(__inference_dropout_4_layer_call_fn_8017

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_7216`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_7329

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï	
`
A__inference_dropout_layer_call_and_return_conditional_losses_7931

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
_
A__inference_dropout_layer_call_and_return_conditional_losses_7160

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7958

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï	
`
A__inference_dropout_layer_call_and_return_conditional_losses_7375

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_dropout_3_layer_call_fn_7990

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7202`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7174

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡I
õ
__inference__wrapped_model_7141
input_1@
.sobolev_network_matmul_readvariableop_resource:9
+sobolev_network_add_readvariableop_resource:B
0sobolev_network_matmul_1_readvariableop_resource:;
-sobolev_network_add_1_readvariableop_resource:B
0sobolev_network_matmul_2_readvariableop_resource:;
-sobolev_network_add_2_readvariableop_resource:B
0sobolev_network_matmul_3_readvariableop_resource:;
-sobolev_network_add_3_readvariableop_resource:B
0sobolev_network_matmul_4_readvariableop_resource:;
-sobolev_network_add_4_readvariableop_resource:B
0sobolev_network_matmul_5_readvariableop_resource:;
-sobolev_network_add_5_readvariableop_resource:B
0sobolev_network_matmul_6_readvariableop_resource:;
-sobolev_network_add_6_readvariableop_resource:
identity¢%sobolev_network/MatMul/ReadVariableOp¢'sobolev_network/MatMul_1/ReadVariableOp¢'sobolev_network/MatMul_2/ReadVariableOp¢'sobolev_network/MatMul_3/ReadVariableOp¢'sobolev_network/MatMul_4/ReadVariableOp¢'sobolev_network/MatMul_5/ReadVariableOp¢'sobolev_network/MatMul_6/ReadVariableOp¢"sobolev_network/add/ReadVariableOp¢$sobolev_network/add_1/ReadVariableOp¢$sobolev_network/add_2/ReadVariableOp¢$sobolev_network/add_3/ReadVariableOp¢$sobolev_network/add_4/ReadVariableOp¢$sobolev_network/add_5/ReadVariableOp¢$sobolev_network/add_6/ReadVariableOp
%sobolev_network/MatMul/ReadVariableOpReadVariableOp.sobolev_network_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
sobolev_network/MatMulMatMulinput_1-sobolev_network/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sobolev_network/add/ReadVariableOpReadVariableOp+sobolev_network_add_readvariableop_resource*
_output_shapes
:*
dtype0
sobolev_network/addAddV2 sobolev_network/MatMul:product:0*sobolev_network/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
sobolev_network/TanhTanhsobolev_network/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 sobolev_network/dropout/IdentityIdentitysobolev_network/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sobolev_network/MatMul_1/ReadVariableOpReadVariableOp0sobolev_network_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0°
sobolev_network/MatMul_1MatMul)sobolev_network/dropout/Identity:output:0/sobolev_network/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sobolev_network/add_1/ReadVariableOpReadVariableOp-sobolev_network_add_1_readvariableop_resource*
_output_shapes
:*
dtype0¢
sobolev_network/add_1AddV2"sobolev_network/MatMul_1:product:0,sobolev_network/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
sobolev_network/Tanh_1Tanhsobolev_network/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
"sobolev_network/dropout_1/IdentityIdentitysobolev_network/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sobolev_network/MatMul_2/ReadVariableOpReadVariableOp0sobolev_network_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0²
sobolev_network/MatMul_2MatMul+sobolev_network/dropout_1/Identity:output:0/sobolev_network/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sobolev_network/add_2/ReadVariableOpReadVariableOp-sobolev_network_add_2_readvariableop_resource*
_output_shapes
:*
dtype0¢
sobolev_network/add_2AddV2"sobolev_network/MatMul_2:product:0,sobolev_network/add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
sobolev_network/SigmoidSigmoidsobolev_network/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
"sobolev_network/dropout_2/IdentityIdentitysobolev_network/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sobolev_network/MatMul_3/ReadVariableOpReadVariableOp0sobolev_network_matmul_3_readvariableop_resource*
_output_shapes

:*
dtype0²
sobolev_network/MatMul_3MatMul+sobolev_network/dropout_2/Identity:output:0/sobolev_network/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sobolev_network/add_3/ReadVariableOpReadVariableOp-sobolev_network_add_3_readvariableop_resource*
_output_shapes
:*
dtype0¢
sobolev_network/add_3AddV2"sobolev_network/MatMul_3:product:0,sobolev_network/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
sobolev_network/Sigmoid_1Sigmoidsobolev_network/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sobolev_network/dropout_3/IdentityIdentitysobolev_network/Sigmoid_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sobolev_network/MatMul_4/ReadVariableOpReadVariableOp0sobolev_network_matmul_4_readvariableop_resource*
_output_shapes

:*
dtype0²
sobolev_network/MatMul_4MatMul+sobolev_network/dropout_3/Identity:output:0/sobolev_network/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sobolev_network/add_4/ReadVariableOpReadVariableOp-sobolev_network_add_4_readvariableop_resource*
_output_shapes
:*
dtype0¢
sobolev_network/add_4AddV2"sobolev_network/MatMul_4:product:0,sobolev_network/add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
sobolev_network/LeakyRelu	LeakyRelusobolev_network/add_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sobolev_network/dropout_4/IdentityIdentity'sobolev_network/LeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sobolev_network/MatMul_5/ReadVariableOpReadVariableOp0sobolev_network_matmul_5_readvariableop_resource*
_output_shapes

:*
dtype0²
sobolev_network/MatMul_5MatMul+sobolev_network/dropout_4/Identity:output:0/sobolev_network/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sobolev_network/add_5/ReadVariableOpReadVariableOp-sobolev_network_add_5_readvariableop_resource*
_output_shapes
:*
dtype0¢
sobolev_network/add_5AddV2"sobolev_network/MatMul_5:product:0,sobolev_network/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
sobolev_network/ReluRelusobolev_network/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sobolev_network/MatMul_6/ReadVariableOpReadVariableOp0sobolev_network_matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0©
sobolev_network/MatMul_6MatMul"sobolev_network/Relu:activations:0/sobolev_network/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sobolev_network/add_6/ReadVariableOpReadVariableOp-sobolev_network_add_6_readvariableop_resource*
_output_shapes
:*
dtype0¢
sobolev_network/add_6AddV2"sobolev_network/MatMul_6:product:0,sobolev_network/add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitysobolev_network/add_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
NoOpNoOp&^sobolev_network/MatMul/ReadVariableOp(^sobolev_network/MatMul_1/ReadVariableOp(^sobolev_network/MatMul_2/ReadVariableOp(^sobolev_network/MatMul_3/ReadVariableOp(^sobolev_network/MatMul_4/ReadVariableOp(^sobolev_network/MatMul_5/ReadVariableOp(^sobolev_network/MatMul_6/ReadVariableOp#^sobolev_network/add/ReadVariableOp%^sobolev_network/add_1/ReadVariableOp%^sobolev_network/add_2/ReadVariableOp%^sobolev_network/add_3/ReadVariableOp%^sobolev_network/add_4/ReadVariableOp%^sobolev_network/add_5/ReadVariableOp%^sobolev_network/add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2N
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
í
a
(__inference_dropout_1_layer_call_fn_7941

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7352o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
a
(__inference_dropout_4_layer_call_fn_8022

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_7283o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_dropout_1_layer_call_fn_7936

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7174`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÕE


I__inference_sobolev_network_layer_call_and_return_conditional_losses_7470
x0
matmul_readvariableop_resource:)
add_readvariableop_resource:2
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:+
add_2_readvariableop_resource:2
 matmul_3_readvariableop_resource:+
add_3_readvariableop_resource:2
 matmul_4_readvariableop_resource:+
add_4_readvariableop_resource:2
 matmul_5_readvariableop_resource:+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢MatMul_3/ReadVariableOp¢MatMul_4/ReadVariableOp¢MatMul_5/ReadVariableOp¢MatMul_6/ReadVariableOp¢add/ReadVariableOp¢add_1/ReadVariableOp¢add_2/ReadVariableOp¢add_3/ReadVariableOp¢add_4/ReadVariableOp¢add_5/ReadVariableOp¢add_6/ReadVariableOp¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCallt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
dropout/StatefulPartitionedCallStatefulPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7375x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_1MatMul(dropout/StatefulPartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall
Tanh_1:y:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7352x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_2MatMul*dropout_1/StatefulPartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
SigmoidSigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallSigmoid:y:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7329x
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_3MatMul*dropout_2/StatefulPartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallSigmoid_1:y:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7306x
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_4MatMul*dropout_3/StatefulPartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype0r
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
	LeakyRelu	LeakyRelu	add_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_7283x
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_5MatMul*dropout_4/StatefulPartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2.
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameX
é
_
&__inference_dropout_layer_call_fn_7914

inputs
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7375o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_7973

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_4_layer_call_and_return_conditional_losses_8039

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Ý
.__inference_sobolev_network_layer_call_fn_7722
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallø
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameX
ñ	
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_7985

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_8000

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
5
Ù
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7812
x0
matmul_readvariableop_resource:)
add_readvariableop_resource:2
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:+
add_2_readvariableop_resource:2
 matmul_3_readvariableop_resource:+
add_3_readvariableop_resource:2
 matmul_4_readvariableop_resource:+
add_4_readvariableop_resource:2
 matmul_5_readvariableop_resource:+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢MatMul_3/ReadVariableOp¢MatMul_4/ReadVariableOp¢MatMul_5/ReadVariableOp¢MatMul_6/ReadVariableOp¢add/ReadVariableOp¢add_1/ReadVariableOp¢add_2/ReadVariableOp¢add_3/ReadVariableOp¢add_4/ReadVariableOp¢add_5/ReadVariableOp¢add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dropout/IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_1MatMuldropout/Identity:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_1/IdentityIdentity
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_2MatMuldropout_1/Identity:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
SigmoidSigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_2/IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_3MatMuldropout_2/Identity:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_3/IdentityIdentitySigmoid_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_4MatMuldropout_3/Identity:output:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype0r
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
	LeakyRelu	LeakyRelu	add_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout_4/IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_5MatMuldropout_4/Identity:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2.
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameX
ñ	
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_8012

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
>
Ù
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7232
x0
matmul_readvariableop_resource:)
add_readvariableop_resource:2
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:+
add_2_readvariableop_resource:2
 matmul_3_readvariableop_resource:+
add_3_readvariableop_resource:2
 matmul_4_readvariableop_resource:+
add_4_readvariableop_resource:2
 matmul_5_readvariableop_resource:+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢MatMul_3/ReadVariableOp¢MatMul_4/ReadVariableOp¢MatMul_5/ReadVariableOp¢MatMul_6/ReadVariableOp¢add/ReadVariableOp¢add_1/ReadVariableOp¢add_2/ReadVariableOp¢add_3/ReadVariableOp¢add_4/ReadVariableOp¢add_5/ReadVariableOp¢add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0d
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
dropout/PartitionedCallPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7160x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_1MatMul dropout/PartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
dropout_1/PartitionedCallPartitionedCall
Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7174x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_2MatMul"dropout_1/PartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
SigmoidSigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
dropout_2/PartitionedCallPartitionedCallSigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7188x
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_3MatMul"dropout_2/PartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
dropout_3/PartitionedCallPartitionedCallSigmoid_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7202x
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_4MatMul"dropout_3/PartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype0r
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
	LeakyRelu	LeakyRelu	add_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dropout_4/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_7216x
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_5MatMul"dropout_4/PartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2.
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameX
çE


I__inference_sobolev_network_layer_call_and_return_conditional_losses_7648
input_10
matmul_readvariableop_resource:)
add_readvariableop_resource:2
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:+
add_2_readvariableop_resource:2
 matmul_3_readvariableop_resource:+
add_3_readvariableop_resource:2
 matmul_4_readvariableop_resource:+
add_4_readvariableop_resource:2
 matmul_5_readvariableop_resource:+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢MatMul_3/ReadVariableOp¢MatMul_4/ReadVariableOp¢MatMul_5/ReadVariableOp¢MatMul_6/ReadVariableOp¢add/ReadVariableOp¢add_1/ReadVariableOp¢add_2/ReadVariableOp¢add_3/ReadVariableOp¢add_4/ReadVariableOp¢add_5/ReadVariableOp¢add_6/ReadVariableOp¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCallt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0j
MatMulMatMulinput_1MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
dropout/StatefulPartitionedCallStatefulPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7375x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_1MatMul(dropout/StatefulPartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall
Tanh_1:y:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7352x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_2MatMul*dropout_1/StatefulPartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
SigmoidSigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCallSigmoid:y:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7329x
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_3MatMul*dropout_2/StatefulPartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
!dropout_3/StatefulPartitionedCallStatefulPartitionedCallSigmoid_1:y:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7306x
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_4MatMul*dropout_3/StatefulPartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype0r
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
	LeakyRelu	LeakyRelu	add_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
!dropout_4/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_7283x
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_5MatMul*dropout_4/StatefulPartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2.
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

D
(__inference_dropout_2_layer_call_fn_7963

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7188`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_7216

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_7306

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7946

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_4_layer_call_and_return_conditional_losses_8027

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_4_layer_call_and_return_conditional_losses_7283

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_7188

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_7202

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
a
(__inference_dropout_2_layer_call_fn_7968

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7329o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
_
A__inference_dropout_layer_call_and_return_conditional_losses_7919

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Ý
.__inference_sobolev_network_layer_call_fn_7755
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallø
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameX

B
&__inference_dropout_layer_call_fn_7909

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7160`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
ã
.__inference_sobolev_network_layer_call_fn_7534
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:
identity¢StatefulPartitionedCallþ
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
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
>
ß
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7591
input_10
matmul_readvariableop_resource:)
add_readvariableop_resource:2
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:+
add_2_readvariableop_resource:2
 matmul_3_readvariableop_resource:+
add_3_readvariableop_resource:2
 matmul_4_readvariableop_resource:+
add_4_readvariableop_resource:2
 matmul_5_readvariableop_resource:+
add_5_readvariableop_resource:2
 matmul_6_readvariableop_resource:+
add_6_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢MatMul_2/ReadVariableOp¢MatMul_3/ReadVariableOp¢MatMul_4/ReadVariableOp¢MatMul_5/ReadVariableOp¢MatMul_6/ReadVariableOp¢add/ReadVariableOp¢add_1/ReadVariableOp¢add_2/ReadVariableOp¢add_3/ReadVariableOp¢add_4/ReadVariableOp¢add_5/ReadVariableOp¢add_6/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0j
MatMulMatMulinput_1MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0l
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
dropout/PartitionedCallPartitionedCallTanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7160x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_1MatMul dropout/PartitionedCall:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0r
add_1AddV2MatMul_1:product:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
dropout_1/PartitionedCallPartitionedCall
Tanh_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7174x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_2MatMul"dropout_1/PartitionedCall:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes
:*
dtype0r
add_2AddV2MatMul_2:product:0add_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
SigmoidSigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
dropout_2/PartitionedCallPartitionedCallSigmoid:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7188x
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_3MatMul"dropout_2/PartitionedCall:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2MatMul_3:product:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
dropout_3/PartitionedCallPartitionedCallSigmoid_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_7202x
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_4MatMul"dropout_3/PartitionedCall:output:0MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_4/ReadVariableOpReadVariableOpadd_4_readvariableop_resource*
_output_shapes
:*
dtype0r
add_4AddV2MatMul_4:product:0add_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
	LeakyRelu	LeakyRelu	add_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dropout_4/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_4_layer_call_and_return_conditional_losses_7216x
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:*
dtype0
MatMul_5MatMul"dropout_4/PartitionedCall:output:0MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:*
dtype0r
add_5AddV2MatMul_5:product:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:*
dtype0y
MatMul_6MatMulRelu:activations:0MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
add_6/ReadVariableOpReadVariableOpadd_6_readvariableop_resource*
_output_shapes
:*
dtype0r
add_6AddV2MatMul_6:product:0add_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_6:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2.
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ý¸
 
 __inference__traced_restore_8366
file_prefix.
assignvariableop_variable_13:,
assignvariableop_1_variable_12:0
assignvariableop_2_variable_11:,
assignvariableop_3_variable_10:/
assignvariableop_4_variable_9:+
assignvariableop_5_variable_8:/
assignvariableop_6_variable_7:+
assignvariableop_7_variable_6:/
assignvariableop_8_variable_5:+
assignvariableop_9_variable_4:0
assignvariableop_10_variable_3:,
assignvariableop_11_variable_2:0
assignvariableop_12_variable_1:*
assignvariableop_13_variable:$
assignvariableop_14_beta_1: $
assignvariableop_15_beta_2: #
assignvariableop_16_decay: +
!assignvariableop_17_learning_rate: '
assignvariableop_18_adam_iter:	 #
assignvariableop_19_total: #
assignvariableop_20_count: 8
&assignvariableop_21_adam_variable_m_13:4
&assignvariableop_22_adam_variable_m_12:8
&assignvariableop_23_adam_variable_m_11:4
&assignvariableop_24_adam_variable_m_10:7
%assignvariableop_25_adam_variable_m_9:3
%assignvariableop_26_adam_variable_m_8:7
%assignvariableop_27_adam_variable_m_7:3
%assignvariableop_28_adam_variable_m_6:7
%assignvariableop_29_adam_variable_m_5:3
%assignvariableop_30_adam_variable_m_4:7
%assignvariableop_31_adam_variable_m_3:3
%assignvariableop_32_adam_variable_m_2:7
%assignvariableop_33_adam_variable_m_1:1
#assignvariableop_34_adam_variable_m:8
&assignvariableop_35_adam_variable_v_13:4
&assignvariableop_36_adam_variable_v_12:8
&assignvariableop_37_adam_variable_v_11:4
&assignvariableop_38_adam_variable_v_10:7
%assignvariableop_39_adam_variable_v_9:3
%assignvariableop_40_adam_variable_v_8:7
%assignvariableop_41_adam_variable_v_7:3
%assignvariableop_42_adam_variable_v_6:7
%assignvariableop_43_adam_variable_v_5:3
%assignvariableop_44_adam_variable_v_4:7
%assignvariableop_45_adam_variable_v_3:3
%assignvariableop_46_adam_variable_v_2:7
%assignvariableop_47_adam_variable_v_1:1
#assignvariableop_48_adam_variable_v:
identity_50¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ª
value B2BW1/.ATTRIBUTES/VARIABLE_VALUEBb1/.ATTRIBUTES/VARIABLE_VALUEBW2/.ATTRIBUTES/VARIABLE_VALUEBb2/.ATTRIBUTES/VARIABLE_VALUEBW3/.ATTRIBUTES/VARIABLE_VALUEBb3/.ATTRIBUTES/VARIABLE_VALUEBW4/.ATTRIBUTES/VARIABLE_VALUEBb4/.ATTRIBUTES/VARIABLE_VALUEBW5/.ATTRIBUTES/VARIABLE_VALUEBb5/.ATTRIBUTES/VARIABLE_VALUEBW6/.ATTRIBUTES/VARIABLE_VALUEBb6/.ATTRIBUTES/VARIABLE_VALUEBW7/.ATTRIBUTES/VARIABLE_VALUEBb7/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9W7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9b7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÔ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Þ
_output_shapesË
È::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_variable_13Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_12Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_11Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_10Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_9Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_8Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_7Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_6Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_5Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_4Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_3Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_variableIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_variable_m_13Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_variable_m_12Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adam_variable_m_11Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_variable_m_10Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_variable_m_9Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_variable_m_8Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_variable_m_7Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_variable_m_6Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_variable_m_5Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_variable_m_4Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_variable_m_3Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_variable_m_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_variable_m_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp#assignvariableop_34_adam_variable_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp&assignvariableop_35_adam_variable_v_13Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_variable_v_12Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp&assignvariableop_37_adam_variable_v_11Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_variable_v_10Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_variable_v_9Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_variable_v_8Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_variable_v_7Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_variable_v_6Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adam_variable_v_5Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_variable_v_4Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp%assignvariableop_45_adam_variable_v_3Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_variable_v_2Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp%assignvariableop_47_adam_variable_v_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp#assignvariableop_48_adam_variable_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: ò
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:³À

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

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

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
Ê
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
è
$trace_0
%trace_1
&trace_2
'trace_32ý
.__inference_sobolev_network_layer_call_fn_7263
.__inference_sobolev_network_layer_call_fn_7722
.__inference_sobolev_network_layer_call_fn_7755
.__inference_sobolev_network_layer_call_fn_7534º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 z$trace_0z%trace_1z&trace_2z'trace_3
Ô
(trace_0
)trace_1
*trace_2
+trace_32é
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7812
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7904
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7591
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7648º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 z(trace_0z)trace_1z*trace_2z+trace_3
ÊBÇ
__inference__wrapped_model_7141input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:2Variable
:2Variable
¼
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator"
_tf_keras_layer
:2Variable
:2Variable
¼
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator"
_tf_keras_layer
:2Variable
:2Variable
¼
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@_random_generator"
_tf_keras_layer
:2Variable
:2Variable
¼
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator"
_tf_keras_layer
:2Variable
:2Variable
¼
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_random_generator"
_tf_keras_layer
:2Variable
:2Variable
?
O	keras_api
P_random_generator"
_tf_keras_layer
:2Variable
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
ë

Xbeta_1

Ybeta_2
	Zdecay
[learning_rate
\iterm	mmmmmmmmmmmmmv	vv v¡v¢v£v¤v¥v¦v§v¨v©vªv«"
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
'
^0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
.__inference_sobolev_network_layer_call_fn_7263input_1"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
õBò
.__inference_sobolev_network_layer_call_fn_7722X"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
õBò
.__inference_sobolev_network_layer_call_fn_7755X"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
ûBø
.__inference_sobolev_network_layer_call_fn_7534input_1"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7812X"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7904X"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7591input_1"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7648input_1"º
±²­
FullArgSpec
args
jself
jX
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
½
dtrace_0
etrace_12
&__inference_dropout_layer_call_fn_7909
&__inference_dropout_layer_call_fn_7914³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zdtrace_0zetrace_1
ó
ftrace_0
gtrace_12¼
A__inference_dropout_layer_call_and_return_conditional_losses_7919
A__inference_dropout_layer_call_and_return_conditional_losses_7931³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zftrace_0zgtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
Á
mtrace_0
ntrace_12
(__inference_dropout_1_layer_call_fn_7936
(__inference_dropout_1_layer_call_fn_7941³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zmtrace_0zntrace_1
÷
otrace_0
ptrace_12À
C__inference_dropout_1_layer_call_and_return_conditional_losses_7946
C__inference_dropout_1_layer_call_and_return_conditional_losses_7958³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0zptrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Á
vtrace_0
wtrace_12
(__inference_dropout_2_layer_call_fn_7963
(__inference_dropout_2_layer_call_fn_7968³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0zwtrace_1
÷
xtrace_0
ytrace_12À
C__inference_dropout_2_layer_call_and_return_conditional_losses_7973
C__inference_dropout_2_layer_call_and_return_conditional_losses_7985³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zxtrace_0zytrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Ã
trace_0
trace_12
(__inference_dropout_3_layer_call_fn_7990
(__inference_dropout_3_layer_call_fn_7995³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
û
trace_0
trace_12À
C__inference_dropout_3_layer_call_and_return_conditional_losses_8000
C__inference_dropout_3_layer_call_and_return_conditional_losses_8012³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
Å
trace_0
trace_12
(__inference_dropout_4_layer_call_fn_8017
(__inference_dropout_4_layer_call_fn_8022³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
û
trace_0
trace_12À
C__inference_dropout_4_layer_call_and_return_conditional_losses_8027
C__inference_dropout_4_layer_call_and_return_conditional_losses_8039³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
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
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
ÉBÆ
"__inference_signature_wrapper_7689input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
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
ëBè
&__inference_dropout_layer_call_fn_7909inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ëBè
&__inference_dropout_layer_call_fn_7914inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
A__inference_dropout_layer_call_and_return_conditional_losses_7919inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
A__inference_dropout_layer_call_and_return_conditional_losses_7931inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
íBê
(__inference_dropout_1_layer_call_fn_7936inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
íBê
(__inference_dropout_1_layer_call_fn_7941inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_1_layer_call_and_return_conditional_losses_7946inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_1_layer_call_and_return_conditional_losses_7958inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
íBê
(__inference_dropout_2_layer_call_fn_7963inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
íBê
(__inference_dropout_2_layer_call_fn_7968inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_2_layer_call_and_return_conditional_losses_7973inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_2_layer_call_and_return_conditional_losses_7985inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
íBê
(__inference_dropout_3_layer_call_fn_7990inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
íBê
(__inference_dropout_3_layer_call_fn_7995inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_3_layer_call_and_return_conditional_losses_8000inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_3_layer_call_and_return_conditional_losses_8012inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
íBê
(__inference_dropout_4_layer_call_fn_8017inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
íBê
(__inference_dropout_4_layer_call_fn_8022inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_4_layer_call_and_return_conditional_losses_8027inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
C__inference_dropout_4_layer_call_and_return_conditional_losses_8039inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/m
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
:2Adam/Variable/v
__inference__wrapped_model_7141w	0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_1_layer_call_and_return_conditional_losses_7946\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_1_layer_call_and_return_conditional_losses_7958\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_1_layer_call_fn_7936O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_1_layer_call_fn_7941O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_2_layer_call_and_return_conditional_losses_7973\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_2_layer_call_and_return_conditional_losses_7985\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_2_layer_call_fn_7963O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_2_layer_call_fn_7968O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_3_layer_call_and_return_conditional_losses_8000\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_3_layer_call_and_return_conditional_losses_8012\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_3_layer_call_fn_7990O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_3_layer_call_fn_7995O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_4_layer_call_and_return_conditional_losses_8027\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_4_layer_call_and_return_conditional_losses_8039\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_4_layer_call_fn_8017O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_4_layer_call_fn_8022O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¡
A__inference_dropout_layer_call_and_return_conditional_losses_7919\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
A__inference_dropout_layer_call_and_return_conditional_losses_7931\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dropout_layer_call_fn_7909O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿy
&__inference_dropout_layer_call_fn_7914O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ©
"__inference_signature_wrapper_7689	;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿÆ
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7591y	@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7648y	@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7812s	:¢7
 ¢

Xÿÿÿÿÿÿÿÿÿ
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ
 À
I__inference_sobolev_network_layer_call_and_return_conditional_losses_7904s	:¢7
 ¢

Xÿÿÿÿÿÿÿÿÿ
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_sobolev_network_layer_call_fn_7263l	@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp "ÿÿÿÿÿÿÿÿÿ
.__inference_sobolev_network_layer_call_fn_7534l	@¢=
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp"ÿÿÿÿÿÿÿÿÿ
.__inference_sobolev_network_layer_call_fn_7722f	:¢7
 ¢

Xÿÿÿÿÿÿÿÿÿ
ª

trainingp "ÿÿÿÿÿÿÿÿÿ
.__inference_sobolev_network_layer_call_fn_7755f	:¢7
 ¢

Xÿÿÿÿÿÿÿÿÿ
ª

trainingp"ÿÿÿÿÿÿÿÿÿ