ó#
ĚŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
ž
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.1-0-gfcc4b966f18šą
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	b*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	b*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
y
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*(
_output_shapes
:*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
}
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*(
_output_shapes
:*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:*
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
}
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:*
dtype0
s
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
l
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes	
:*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel

$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*(
_output_shapes
:*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel

$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*(
_output_shapes
:*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_13/kernel

$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*(
_output_shapes
:*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:*
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_15/kernel

$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*(
_output_shapes
:*
dtype0
u
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
n
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes	
:*
dtype0

conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_16/kernel

$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*(
_output_shapes
:*
dtype0
u
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_16/bias
n
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes	
:*
dtype0

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
:*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ć
valueťBˇ BŻ
ž
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer-16
layer_with_weights-5
layer-17
layer-18
layer-19
layer_with_weights-6
layer-20
layer-21
layer-22
layer-23
layer_with_weights-7
layer-24
layer-25
layer-26
layer_with_weights-8
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-9
 layer-31
!layer-32
"layer-33
#layer_with_weights-10
#layer-34
$layer-35
%layer-36
&layer-37
'layer_with_weights-11
'layer-38
(layer-39
)layer-40
*layer_with_weights-12
*layer-41
+layer-42
,layer-43
-layer_with_weights-13
-layer-44
.trainable_variables
/	variables
0regularization_losses
1	keras_api
2
signatures
 
h

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
R
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
R
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
R
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
R
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
R
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
h

]kernel
^bias
_trainable_variables
`	variables
aregularization_losses
b	keras_api
R
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
R
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
h

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
R
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
R
utrainable_variables
v	variables
wregularization_losses
x	keras_api
R
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
k

}kernel
~bias
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
n
kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
n
kernel
	bias
trainable_variables
 	variables
Ąregularization_losses
˘	keras_api
V
Łtrainable_variables
¤	variables
Ľregularization_losses
Ś	keras_api
V
§trainable_variables
¨	variables
Šregularization_losses
Ş	keras_api
n
Ťkernel
	Źbias
­trainable_variables
Ž	variables
Żregularization_losses
°	keras_api
V
ątrainable_variables
˛	variables
łregularization_losses
´	keras_api
V
ľtrainable_variables
ś	variables
ˇregularization_losses
¸	keras_api
V
štrainable_variables
ş	variables
ťregularization_losses
ź	keras_api
n
˝kernel
	žbias
żtrainable_variables
Ŕ	variables
Áregularization_losses
Â	keras_api
V
Ătrainable_variables
Ä	variables
Ĺregularization_losses
Ć	keras_api
V
Çtrainable_variables
Č	variables
Éregularization_losses
Ę	keras_api
n
Ëkernel
	Ěbias
Ítrainable_variables
Î	variables
Ďregularization_losses
Đ	keras_api
V
Ńtrainable_variables
Ň	variables
Óregularization_losses
Ô	keras_api
V
Őtrainable_variables
Ö	variables
×regularization_losses
Ř	keras_api
V
Ůtrainable_variables
Ú	variables
Űregularization_losses
Ü	keras_api
n
Ýkernel
	Ţbias
ßtrainable_variables
ŕ	variables
áregularization_losses
â	keras_api
V
ătrainable_variables
ä	variables
ĺregularization_losses
ć	keras_api
V
çtrainable_variables
č	variables
éregularization_losses
ę	keras_api
n
ëkernel
	ěbias
ítrainable_variables
î	variables
ďregularization_losses
đ	keras_api
V
ńtrainable_variables
ň	variables
óregularization_losses
ô	keras_api
V
őtrainable_variables
ö	variables
÷regularization_losses
ř	keras_api
n
ůkernel
	úbias
űtrainable_variables
ü	variables
ýregularization_losses
ţ	keras_api
ć
30
41
=2
>3
K4
L5
]6
^7
k8
l9
}10
~11
12
13
14
15
Ť16
Ź17
˝18
ž19
Ë20
Ě21
Ý22
Ţ23
ë24
ě25
ů26
ú27
ć
30
41
=2
>3
K4
L5
]6
^7
k8
l9
}10
~11
12
13
14
15
Ť16
Ź17
˝18
ž19
Ë20
Ě21
Ý22
Ţ23
ë24
ě25
ů26
ú27
 
˛
.trainable_variables
 ˙layer_regularization_losses
layers
layer_metrics
non_trainable_variables
metrics
/	variables
0regularization_losses
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
˛
5trainable_variables
 layer_regularization_losses
6	variables
layer_metrics
non_trainable_variables
metrics
layers
7regularization_losses
 
 
 
˛
9trainable_variables
 layer_regularization_losses
:	variables
layer_metrics
non_trainable_variables
metrics
layers
;regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
˛
?trainable_variables
 layer_regularization_losses
@	variables
layer_metrics
non_trainable_variables
metrics
layers
Aregularization_losses
 
 
 
˛
Ctrainable_variables
 layer_regularization_losses
D	variables
layer_metrics
non_trainable_variables
metrics
layers
Eregularization_losses
 
 
 
˛
Gtrainable_variables
 layer_regularization_losses
H	variables
layer_metrics
non_trainable_variables
metrics
layers
Iregularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
˛
Mtrainable_variables
 layer_regularization_losses
N	variables
layer_metrics
non_trainable_variables
 metrics
Ąlayers
Oregularization_losses
 
 
 
˛
Qtrainable_variables
 ˘layer_regularization_losses
R	variables
Łlayer_metrics
¤non_trainable_variables
Ľmetrics
Ślayers
Sregularization_losses
 
 
 
˛
Utrainable_variables
 §layer_regularization_losses
V	variables
¨layer_metrics
Šnon_trainable_variables
Şmetrics
Ťlayers
Wregularization_losses
 
 
 
˛
Ytrainable_variables
 Źlayer_regularization_losses
Z	variables
­layer_metrics
Žnon_trainable_variables
Żmetrics
°layers
[regularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

]0
^1
 
˛
_trainable_variables
 ąlayer_regularization_losses
`	variables
˛layer_metrics
łnon_trainable_variables
´metrics
ľlayers
aregularization_losses
 
 
 
˛
ctrainable_variables
 ślayer_regularization_losses
d	variables
ˇlayer_metrics
¸non_trainable_variables
šmetrics
şlayers
eregularization_losses
 
 
 
˛
gtrainable_variables
 ťlayer_regularization_losses
h	variables
źlayer_metrics
˝non_trainable_variables
žmetrics
żlayers
iregularization_losses
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

k0
l1
 
˛
mtrainable_variables
 Ŕlayer_regularization_losses
n	variables
Álayer_metrics
Ânon_trainable_variables
Ămetrics
Älayers
oregularization_losses
 
 
 
˛
qtrainable_variables
 Ĺlayer_regularization_losses
r	variables
Ćlayer_metrics
Çnon_trainable_variables
Čmetrics
Élayers
sregularization_losses
 
 
 
˛
utrainable_variables
 Ęlayer_regularization_losses
v	variables
Ëlayer_metrics
Ěnon_trainable_variables
Ímetrics
Îlayers
wregularization_losses
 
 
 
˛
ytrainable_variables
 Ďlayer_regularization_losses
z	variables
Đlayer_metrics
Ńnon_trainable_variables
Ňmetrics
Ólayers
{regularization_losses
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1

}0
~1
 
´
trainable_variables
 Ôlayer_regularization_losses
	variables
Őlayer_metrics
Önon_trainable_variables
×metrics
Řlayers
regularization_losses
 
 
 
ľ
trainable_variables
 Ůlayer_regularization_losses
	variables
Úlayer_metrics
Űnon_trainable_variables
Ümetrics
Ýlayers
regularization_losses
 
 
 
ľ
trainable_variables
 Ţlayer_regularization_losses
	variables
ßlayer_metrics
ŕnon_trainable_variables
ámetrics
âlayers
regularization_losses
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
ľ
trainable_variables
 ălayer_regularization_losses
	variables
älayer_metrics
ĺnon_trainable_variables
ćmetrics
çlayers
regularization_losses
 
 
 
ľ
trainable_variables
 člayer_regularization_losses
	variables
élayer_metrics
ęnon_trainable_variables
ëmetrics
ělayers
regularization_losses
 
 
 
ľ
trainable_variables
 ílayer_regularization_losses
	variables
îlayer_metrics
ďnon_trainable_variables
đmetrics
ńlayers
regularization_losses
 
 
 
ľ
trainable_variables
 ňlayer_regularization_losses
	variables
ólayer_metrics
ônon_trainable_variables
őmetrics
ölayers
regularization_losses
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
ľ
trainable_variables
 ÷layer_regularization_losses
 	variables
řlayer_metrics
ůnon_trainable_variables
úmetrics
űlayers
Ąregularization_losses
 
 
 
ľ
Łtrainable_variables
 ülayer_regularization_losses
¤	variables
ýlayer_metrics
ţnon_trainable_variables
˙metrics
layers
Ľregularization_losses
 
 
 
ľ
§trainable_variables
 layer_regularization_losses
¨	variables
layer_metrics
non_trainable_variables
metrics
layers
Šregularization_losses
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

Ť0
Ź1

Ť0
Ź1
 
ľ
­trainable_variables
 layer_regularization_losses
Ž	variables
layer_metrics
non_trainable_variables
metrics
layers
Żregularization_losses
 
 
 
ľ
ątrainable_variables
 layer_regularization_losses
˛	variables
layer_metrics
non_trainable_variables
metrics
layers
łregularization_losses
 
 
 
ľ
ľtrainable_variables
 layer_regularization_losses
ś	variables
layer_metrics
non_trainable_variables
metrics
layers
ˇregularization_losses
 
 
 
ľ
štrainable_variables
 layer_regularization_losses
ş	variables
layer_metrics
non_trainable_variables
metrics
layers
ťregularization_losses
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

˝0
ž1

˝0
ž1
 
ľ
żtrainable_variables
 layer_regularization_losses
Ŕ	variables
layer_metrics
non_trainable_variables
metrics
layers
Áregularization_losses
 
 
 
ľ
Ătrainable_variables
 layer_regularization_losses
Ä	variables
 layer_metrics
Ąnon_trainable_variables
˘metrics
Łlayers
Ĺregularization_losses
 
 
 
ľ
Çtrainable_variables
 ¤layer_regularization_losses
Č	variables
Ľlayer_metrics
Śnon_trainable_variables
§metrics
¨layers
Éregularization_losses
][
VARIABLE_VALUEconv2d_13/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_13/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Ë0
Ě1

Ë0
Ě1
 
ľ
Ítrainable_variables
 Šlayer_regularization_losses
Î	variables
Şlayer_metrics
Ťnon_trainable_variables
Źmetrics
­layers
Ďregularization_losses
 
 
 
ľ
Ńtrainable_variables
 Žlayer_regularization_losses
Ň	variables
Żlayer_metrics
°non_trainable_variables
ąmetrics
˛layers
Óregularization_losses
 
 
 
ľ
Őtrainable_variables
 łlayer_regularization_losses
Ö	variables
´layer_metrics
ľnon_trainable_variables
śmetrics
ˇlayers
×regularization_losses
 
 
 
ľ
Ůtrainable_variables
 ¸layer_regularization_losses
Ú	variables
šlayer_metrics
şnon_trainable_variables
ťmetrics
źlayers
Űregularization_losses
][
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Ý0
Ţ1

Ý0
Ţ1
 
ľ
ßtrainable_variables
 ˝layer_regularization_losses
ŕ	variables
žlayer_metrics
żnon_trainable_variables
Ŕmetrics
Álayers
áregularization_losses
 
 
 
ľ
ătrainable_variables
 Âlayer_regularization_losses
ä	variables
Ălayer_metrics
Änon_trainable_variables
Ĺmetrics
Ćlayers
ĺregularization_losses
 
 
 
ľ
çtrainable_variables
 Çlayer_regularization_losses
č	variables
Člayer_metrics
Énon_trainable_variables
Ęmetrics
Ëlayers
éregularization_losses
][
VARIABLE_VALUEconv2d_16/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_16/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

ë0
ě1

ë0
ě1
 
ľ
ítrainable_variables
 Ělayer_regularization_losses
î	variables
Ílayer_metrics
Înon_trainable_variables
Ďmetrics
Đlayers
ďregularization_losses
 
 
 
ľ
ńtrainable_variables
 Ńlayer_regularization_losses
ň	variables
Ňlayer_metrics
Ónon_trainable_variables
Ômetrics
Őlayers
óregularization_losses
 
 
 
ľ
őtrainable_variables
 Ölayer_regularization_losses
ö	variables
×layer_metrics
Řnon_trainable_variables
Ůmetrics
Úlayers
÷regularization_losses
][
VARIABLE_VALUEconv2d_17/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_17/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

ů0
ú1

ů0
ú1
 
ľ
űtrainable_variables
 Űlayer_regularization_losses
ü	variables
Ülayer_metrics
Ýnon_trainable_variables
Ţmetrics
ßlayers
ýregularization_losses
 
Ţ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b*
dtype0*
shape:˙˙˙˙˙˙˙˙˙b
ş
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_6187311
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
­

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_6188788
ř
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_6188882ŹÖ
°
f
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_6188406

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

c
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_6188146
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yb
powPowdatapow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/yl
addAddV2Mean:output:0add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_valuea
SqrtSqrtclip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sqrth
truedivRealDivdataSqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivh
IdentityIdentitytruediv:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:V R
0
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Î

+__inference_conv2d_17_layer_call_fn_6188681

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_61867962
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_6186142

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_6186383
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
	
Ž
F__inference_conv2d_16_layer_call_and_return_conditional_losses_6188621

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž
­
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6188121

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ

+__inference_conv2d_12_layer_call_fn_6188480

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_61865422
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ž
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6188521

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_6188306

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
đ	
Ž
F__inference_conv2d_17_layer_call_and_return_conditional_losses_6186796

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpľ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Tanhv
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ą
g
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_6186777

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_6186510
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Đ

+__inference_conv2d_13_layer_call_fn_6188530

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_61866052
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6186161

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_6186256
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Í
c
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_6188496
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Í
c
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_6188346
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata


/__inference_functional_19_layer_call_fn_6188023

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_functional_19_layer_call_and_return_conditional_losses_61871892
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_6186269

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6185928

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulŐ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

a
G__inference_pixel_norm_layer_call_and_return_conditional_losses_6186066
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yb
powPowdatapow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/yl
addAddV2Mean:output:0add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_valuea
SqrtSqrtclip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sqrth
truedivRealDivdataSqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivh
IdentityIdentitytruediv:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:V R
0
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namedata
­
ž	
J__inference_functional_19_layer_call_and_return_conditional_losses_6187024

inputs
dense_6186923
dense_6186925
conv2d_6186929
conv2d_6186931
conv2d_1_6186936
conv2d_1_6186938
conv2d_3_6186944
conv2d_3_6186946
conv2d_4_6186951
conv2d_4_6186953
conv2d_6_6186959
conv2d_6_6186961
conv2d_7_6186966
conv2d_7_6186968
conv2d_9_6186974
conv2d_9_6186976
conv2d_10_6186981
conv2d_10_6186983
conv2d_12_6186989
conv2d_12_6186991
conv2d_13_6186996
conv2d_13_6186998
conv2d_15_6187004
conv2d_15_6187006
conv2d_16_6187011
conv2d_16_6187013
conv2d_17_6187018
conv2d_17_6187020
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘!conv2d_10/StatefulPartitionedCall˘!conv2d_12/StatefulPartitionedCall˘!conv2d_13/StatefulPartitionedCall˘!conv2d_15/StatefulPartitionedCall˘!conv2d_16/StatefulPartitionedCall˘!conv2d_17/StatefulPartitionedCall˘ conv2d_3/StatefulPartitionedCall˘ conv2d_4/StatefulPartitionedCall˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_9/StatefulPartitionedCall˘dense/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6186923dense_6186925*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_61859862
dense/StatefulPartitionedCallţ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_61860162
reshape/PartitionedCallł
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_6186929conv2d_6186931*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_61860342 
conv2d/StatefulPartitionedCall
pixel_norm/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_pixel_norm_layer_call_and_return_conditional_losses_61860662
pixel_norm/PartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall#pixel_norm/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_61860792
leaky_re_lu/PartitionedCallÁ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv2d_1_6186936conv2d_1_6186938*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_61860972"
 conv2d_1/StatefulPartitionedCall
pixel_norm_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_61861292
pixel_norm_1/PartitionedCall
leaky_re_lu_1/PartitionedCallPartitionedCall%pixel_norm_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_61861422
leaky_re_lu_1/PartitionedCall˘
up_sampling2d/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_61858902
up_sampling2d/PartitionedCallŐ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_6186944conv2d_3_6186946*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_61861612"
 conv2d_3/StatefulPartitionedCall˘
pixel_norm_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_61861932
pixel_norm_2/PartitionedCallĄ
leaky_re_lu_2/PartitionedCallPartitionedCall%pixel_norm_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_61862062
leaky_re_lu_2/PartitionedCallŐ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv2d_4_6186951conv2d_4_6186953*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_61862242"
 conv2d_4/StatefulPartitionedCall˘
pixel_norm_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_61862562
pixel_norm_3/PartitionedCallĄ
leaky_re_lu_3/PartitionedCallPartitionedCall%pixel_norm_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_61862692
leaky_re_lu_3/PartitionedCall¨
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_61859092!
up_sampling2d_1/PartitionedCall×
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_6186959conv2d_6_6186961*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_61862882"
 conv2d_6/StatefulPartitionedCall˘
pixel_norm_4/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_61863202
pixel_norm_4/PartitionedCallĄ
leaky_re_lu_4/PartitionedCallPartitionedCall%pixel_norm_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_61863332
leaky_re_lu_4/PartitionedCallŐ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv2d_7_6186966conv2d_7_6186968*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_61863512"
 conv2d_7/StatefulPartitionedCall˘
pixel_norm_5/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_61863832
pixel_norm_5/PartitionedCallĄ
leaky_re_lu_5/PartitionedCallPartitionedCall%pixel_norm_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_61863962
leaky_re_lu_5/PartitionedCall¨
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_61859282!
up_sampling2d_2/PartitionedCall×
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_9_6186974conv2d_9_6186976*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_61864152"
 conv2d_9/StatefulPartitionedCall˘
pixel_norm_6/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_61864472
pixel_norm_6/PartitionedCallĄ
leaky_re_lu_6/PartitionedCallPartitionedCall%pixel_norm_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_61864602
leaky_re_lu_6/PartitionedCallÚ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0conv2d_10_6186981conv2d_10_6186983*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_61864782#
!conv2d_10/StatefulPartitionedCallŁ
pixel_norm_7/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_61865102
pixel_norm_7/PartitionedCallĄ
leaky_re_lu_7/PartitionedCallPartitionedCall%pixel_norm_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_61865232
leaky_re_lu_7/PartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_61859472!
up_sampling2d_3/PartitionedCallÜ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_12_6186989conv2d_12_6186991*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_61865422#
!conv2d_12/StatefulPartitionedCallŁ
pixel_norm_8/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_61865742
pixel_norm_8/PartitionedCallĄ
leaky_re_lu_8/PartitionedCallPartitionedCall%pixel_norm_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_61865872
leaky_re_lu_8/PartitionedCallÚ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0conv2d_13_6186996conv2d_13_6186998*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_61866052#
!conv2d_13/StatefulPartitionedCallŁ
pixel_norm_9/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_61866372
pixel_norm_9/PartitionedCallĄ
leaky_re_lu_9/PartitionedCallPartitionedCall%pixel_norm_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_61866502
leaky_re_lu_9/PartitionedCall¨
up_sampling2d_4/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_61859662!
up_sampling2d_4/PartitionedCallÜ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_15_6187004conv2d_15_6187006*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_61866692#
!conv2d_15/StatefulPartitionedCallŚ
pixel_norm_10/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_61867012
pixel_norm_10/PartitionedCallĽ
leaky_re_lu_10/PartitionedCallPartitionedCall&pixel_norm_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_61867142 
leaky_re_lu_10/PartitionedCallŰ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0conv2d_16_6187011conv2d_16_6187013*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_61867322#
!conv2d_16/StatefulPartitionedCallŚ
pixel_norm_11/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_61867642
pixel_norm_11/PartitionedCallĽ
leaky_re_lu_11/PartitionedCallPartitionedCall&pixel_norm_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_61867772 
leaky_re_lu_11/PartitionedCallÚ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0conv2d_17_6187018conv2d_17_6187020*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_61867962#
!conv2d_17/StatefulPartitionedCall
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
Í

*__inference_conv2d_4_layer_call_fn_6188230

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_61862242
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ

+__inference_conv2d_16_layer_call_fn_6188630

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_61867322
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

h
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6185966

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulŐ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_6186447
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

K
/__inference_leaky_re_lu_4_layer_call_fn_6188311

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_61863332
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
f
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_6188156

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í

*__inference_conv2d_6_layer_call_fn_6188280

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_61862882
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_6188506

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

H
.__inference_pixel_norm_2_layer_call_fn_6188201
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_61861932
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

L
0__inference_leaky_re_lu_10_layer_call_fn_6188611

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_61867142
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6188321

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_6186587

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž
K
/__inference_up_sampling2d_layer_call_fn_6185896

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_61858902
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

K
/__inference_leaky_re_lu_5_layer_call_fn_6188361

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_61863962
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Çź
Ć
"__inference__wrapped_model_6185877
input_16
2functional_19_dense_matmul_readvariableop_resource7
3functional_19_dense_biasadd_readvariableop_resource7
3functional_19_conv2d_conv2d_readvariableop_resource8
4functional_19_conv2d_biasadd_readvariableop_resource9
5functional_19_conv2d_1_conv2d_readvariableop_resource:
6functional_19_conv2d_1_biasadd_readvariableop_resource9
5functional_19_conv2d_3_conv2d_readvariableop_resource:
6functional_19_conv2d_3_biasadd_readvariableop_resource9
5functional_19_conv2d_4_conv2d_readvariableop_resource:
6functional_19_conv2d_4_biasadd_readvariableop_resource9
5functional_19_conv2d_6_conv2d_readvariableop_resource:
6functional_19_conv2d_6_biasadd_readvariableop_resource9
5functional_19_conv2d_7_conv2d_readvariableop_resource:
6functional_19_conv2d_7_biasadd_readvariableop_resource9
5functional_19_conv2d_9_conv2d_readvariableop_resource:
6functional_19_conv2d_9_biasadd_readvariableop_resource:
6functional_19_conv2d_10_conv2d_readvariableop_resource;
7functional_19_conv2d_10_biasadd_readvariableop_resource:
6functional_19_conv2d_12_conv2d_readvariableop_resource;
7functional_19_conv2d_12_biasadd_readvariableop_resource:
6functional_19_conv2d_13_conv2d_readvariableop_resource;
7functional_19_conv2d_13_biasadd_readvariableop_resource:
6functional_19_conv2d_15_conv2d_readvariableop_resource;
7functional_19_conv2d_15_biasadd_readvariableop_resource:
6functional_19_conv2d_16_conv2d_readvariableop_resource;
7functional_19_conv2d_16_biasadd_readvariableop_resource:
6functional_19_conv2d_17_conv2d_readvariableop_resource;
7functional_19_conv2d_17_biasadd_readvariableop_resource
identityĘ
)functional_19/dense/MatMul/ReadVariableOpReadVariableOp2functional_19_dense_matmul_readvariableop_resource*
_output_shapes
:	b*
dtype02+
)functional_19/dense/MatMul/ReadVariableOpą
functional_19/dense/MatMulMatMulinput_11functional_19/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/dense/MatMulÉ
*functional_19/dense/BiasAdd/ReadVariableOpReadVariableOp3functional_19_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_19/dense/BiasAdd/ReadVariableOpŇ
functional_19/dense/BiasAddBiasAdd$functional_19/dense/MatMul:product:02functional_19/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/dense/BiasAdd
functional_19/reshape/ShapeShape$functional_19/dense/BiasAdd:output:0*
T0*
_output_shapes
:2
functional_19/reshape/Shape 
)functional_19/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)functional_19/reshape/strided_slice/stack¤
+functional_19/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_19/reshape/strided_slice/stack_1¤
+functional_19/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_19/reshape/strided_slice/stack_2ć
#functional_19/reshape/strided_sliceStridedSlice$functional_19/reshape/Shape:output:02functional_19/reshape/strided_slice/stack:output:04functional_19/reshape/strided_slice/stack_1:output:04functional_19/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#functional_19/reshape/strided_slice
%functional_19/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_19/reshape/Reshape/shape/1
%functional_19/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_19/reshape/Reshape/shape/2
%functional_19/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2'
%functional_19/reshape/Reshape/shape/3ž
#functional_19/reshape/Reshape/shapePack,functional_19/reshape/strided_slice:output:0.functional_19/reshape/Reshape/shape/1:output:0.functional_19/reshape/Reshape/shape/2:output:0.functional_19/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#functional_19/reshape/Reshape/shapeŘ
functional_19/reshape/ReshapeReshape$functional_19/dense/BiasAdd:output:0,functional_19/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/reshape/ReshapeÖ
*functional_19/conv2d/Conv2D/ReadVariableOpReadVariableOp3functional_19_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*functional_19/conv2d/Conv2D/ReadVariableOp
functional_19/conv2d/Conv2DConv2D&functional_19/reshape/Reshape:output:02functional_19/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
functional_19/conv2d/Conv2DĚ
+functional_19/conv2d/BiasAdd/ReadVariableOpReadVariableOp4functional_19_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_19/conv2d/BiasAdd/ReadVariableOpÝ
functional_19/conv2d/BiasAddBiasAdd$functional_19/conv2d/Conv2D:output:03functional_19/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/conv2d/BiasAdd
functional_19/pixel_norm/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
functional_19/pixel_norm/pow/yÎ
functional_19/pixel_norm/powPow%functional_19/conv2d/BiasAdd:output:0'functional_19/pixel_norm/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/pixel_norm/pow­
/functional_19/pixel_norm/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙21
/functional_19/pixel_norm/Mean/reduction_indicesí
functional_19/pixel_norm/MeanMean functional_19/pixel_norm/pow:z:08functional_19/pixel_norm/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
functional_19/pixel_norm/Mean
functional_19/pixel_norm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22 
functional_19/pixel_norm/add/yĐ
functional_19/pixel_norm/addAddV2&functional_19/pixel_norm/Mean:output:0'functional_19/pixel_norm/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/pixel_norm/add
functional_19/pixel_norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
functional_19/pixel_norm/Const
 functional_19/pixel_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2"
 functional_19/pixel_norm/Const_1ň
.functional_19/pixel_norm/clip_by_value/MinimumMinimum functional_19/pixel_norm/add:z:0)functional_19/pixel_norm/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙20
.functional_19/pixel_norm/clip_by_value/Minimumň
&functional_19/pixel_norm/clip_by_valueMaximum2functional_19/pixel_norm/clip_by_value/Minimum:z:0'functional_19/pixel_norm/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2(
&functional_19/pixel_norm/clip_by_valueŹ
functional_19/pixel_norm/SqrtSqrt*functional_19/pixel_norm/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/pixel_norm/SqrtÔ
 functional_19/pixel_norm/truedivRealDiv%functional_19/conv2d/BiasAdd:output:0!functional_19/pixel_norm/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 functional_19/pixel_norm/truedivŻ
#functional_19/leaky_re_lu/LeakyRelu	LeakyRelu$functional_19/pixel_norm/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#functional_19/leaky_re_lu/LeakyReluÜ
,functional_19/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5functional_19_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,functional_19/conv2d_1/Conv2D/ReadVariableOp
functional_19/conv2d_1/Conv2DConv2D1functional_19/leaky_re_lu/LeakyRelu:activations:04functional_19/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
functional_19/conv2d_1/Conv2DŇ
-functional_19/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6functional_19_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-functional_19/conv2d_1/BiasAdd/ReadVariableOpĺ
functional_19/conv2d_1/BiasAddBiasAdd&functional_19/conv2d_1/Conv2D:output:05functional_19/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/conv2d_1/BiasAdd
 functional_19/pixel_norm_1/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_1/pow/yÖ
functional_19/pixel_norm_1/powPow'functional_19/conv2d_1/BiasAdd:output:0)functional_19/pixel_norm_1/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_1/pową
1functional_19/pixel_norm_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_1/Mean/reduction_indiceső
functional_19/pixel_norm_1/MeanMean"functional_19/pixel_norm_1/pow:z:0:functional_19/pixel_norm_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2!
functional_19/pixel_norm_1/Mean
 functional_19/pixel_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_1/add/yŘ
functional_19/pixel_norm_1/addAddV2(functional_19/pixel_norm_1/Mean:output:0)functional_19/pixel_norm_1/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_1/add
 functional_19/pixel_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_1/Const
"functional_19/pixel_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_1/Const_1ú
0functional_19/pixel_norm_1/clip_by_value/MinimumMinimum"functional_19/pixel_norm_1/add:z:0+functional_19/pixel_norm_1/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙22
0functional_19/pixel_norm_1/clip_by_value/Minimumú
(functional_19/pixel_norm_1/clip_by_valueMaximum4functional_19/pixel_norm_1/clip_by_value/Minimum:z:0)functional_19/pixel_norm_1/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(functional_19/pixel_norm_1/clip_by_value˛
functional_19/pixel_norm_1/SqrtSqrt,functional_19/pixel_norm_1/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_1/SqrtÜ
"functional_19/pixel_norm_1/truedivRealDiv'functional_19/conv2d_1/BiasAdd:output:0#functional_19/pixel_norm_1/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"functional_19/pixel_norm_1/truedivľ
%functional_19/leaky_re_lu_1/LeakyRelu	LeakyRelu&functional_19/pixel_norm_1/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%functional_19/leaky_re_lu_1/LeakyReluŠ
!functional_19/up_sampling2d/ShapeShape3functional_19/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2#
!functional_19/up_sampling2d/ShapeŹ
/functional_19/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/functional_19/up_sampling2d/strided_slice/stack°
1functional_19/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1functional_19/up_sampling2d/strided_slice/stack_1°
1functional_19/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1functional_19/up_sampling2d/strided_slice/stack_2ö
)functional_19/up_sampling2d/strided_sliceStridedSlice*functional_19/up_sampling2d/Shape:output:08functional_19/up_sampling2d/strided_slice/stack:output:0:functional_19/up_sampling2d/strided_slice/stack_1:output:0:functional_19/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)functional_19/up_sampling2d/strided_slice
!functional_19/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!functional_19/up_sampling2d/ConstÎ
functional_19/up_sampling2d/mulMul2functional_19/up_sampling2d/strided_slice:output:0*functional_19/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2!
functional_19/up_sampling2d/mulź
8functional_19/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor3functional_19/leaky_re_lu_1/LeakyRelu:activations:0#functional_19/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2:
8functional_19/up_sampling2d/resize/ResizeNearestNeighborÜ
,functional_19/conv2d_3/Conv2D/ReadVariableOpReadVariableOp5functional_19_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,functional_19/conv2d_3/Conv2D/ReadVariableOpŹ
functional_19/conv2d_3/Conv2DConv2DIfunctional_19/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:04functional_19/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
functional_19/conv2d_3/Conv2DŇ
-functional_19/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp6functional_19_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-functional_19/conv2d_3/BiasAdd/ReadVariableOpĺ
functional_19/conv2d_3/BiasAddBiasAdd&functional_19/conv2d_3/Conv2D:output:05functional_19/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/conv2d_3/BiasAdd
 functional_19/pixel_norm_2/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_2/pow/yÖ
functional_19/pixel_norm_2/powPow'functional_19/conv2d_3/BiasAdd:output:0)functional_19/pixel_norm_2/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_2/pową
1functional_19/pixel_norm_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_2/Mean/reduction_indiceső
functional_19/pixel_norm_2/MeanMean"functional_19/pixel_norm_2/pow:z:0:functional_19/pixel_norm_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2!
functional_19/pixel_norm_2/Mean
 functional_19/pixel_norm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_2/add/yŘ
functional_19/pixel_norm_2/addAddV2(functional_19/pixel_norm_2/Mean:output:0)functional_19/pixel_norm_2/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_2/add
 functional_19/pixel_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_2/Const
"functional_19/pixel_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_2/Const_1ú
0functional_19/pixel_norm_2/clip_by_value/MinimumMinimum"functional_19/pixel_norm_2/add:z:0+functional_19/pixel_norm_2/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙22
0functional_19/pixel_norm_2/clip_by_value/Minimumú
(functional_19/pixel_norm_2/clip_by_valueMaximum4functional_19/pixel_norm_2/clip_by_value/Minimum:z:0)functional_19/pixel_norm_2/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(functional_19/pixel_norm_2/clip_by_value˛
functional_19/pixel_norm_2/SqrtSqrt,functional_19/pixel_norm_2/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_2/SqrtÜ
"functional_19/pixel_norm_2/truedivRealDiv'functional_19/conv2d_3/BiasAdd:output:0#functional_19/pixel_norm_2/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"functional_19/pixel_norm_2/truedivľ
%functional_19/leaky_re_lu_2/LeakyRelu	LeakyRelu&functional_19/pixel_norm_2/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%functional_19/leaky_re_lu_2/LeakyReluÜ
,functional_19/conv2d_4/Conv2D/ReadVariableOpReadVariableOp5functional_19_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,functional_19/conv2d_4/Conv2D/ReadVariableOp
functional_19/conv2d_4/Conv2DConv2D3functional_19/leaky_re_lu_2/LeakyRelu:activations:04functional_19/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
functional_19/conv2d_4/Conv2DŇ
-functional_19/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp6functional_19_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-functional_19/conv2d_4/BiasAdd/ReadVariableOpĺ
functional_19/conv2d_4/BiasAddBiasAdd&functional_19/conv2d_4/Conv2D:output:05functional_19/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/conv2d_4/BiasAdd
 functional_19/pixel_norm_3/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_3/pow/yÖ
functional_19/pixel_norm_3/powPow'functional_19/conv2d_4/BiasAdd:output:0)functional_19/pixel_norm_3/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_3/pową
1functional_19/pixel_norm_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_3/Mean/reduction_indiceső
functional_19/pixel_norm_3/MeanMean"functional_19/pixel_norm_3/pow:z:0:functional_19/pixel_norm_3/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2!
functional_19/pixel_norm_3/Mean
 functional_19/pixel_norm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_3/add/yŘ
functional_19/pixel_norm_3/addAddV2(functional_19/pixel_norm_3/Mean:output:0)functional_19/pixel_norm_3/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_3/add
 functional_19/pixel_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_3/Const
"functional_19/pixel_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_3/Const_1ú
0functional_19/pixel_norm_3/clip_by_value/MinimumMinimum"functional_19/pixel_norm_3/add:z:0+functional_19/pixel_norm_3/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙22
0functional_19/pixel_norm_3/clip_by_value/Minimumú
(functional_19/pixel_norm_3/clip_by_valueMaximum4functional_19/pixel_norm_3/clip_by_value/Minimum:z:0)functional_19/pixel_norm_3/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(functional_19/pixel_norm_3/clip_by_value˛
functional_19/pixel_norm_3/SqrtSqrt,functional_19/pixel_norm_3/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_3/SqrtÜ
"functional_19/pixel_norm_3/truedivRealDiv'functional_19/conv2d_4/BiasAdd:output:0#functional_19/pixel_norm_3/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"functional_19/pixel_norm_3/truedivľ
%functional_19/leaky_re_lu_3/LeakyRelu	LeakyRelu&functional_19/pixel_norm_3/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%functional_19/leaky_re_lu_3/LeakyRelu­
#functional_19/up_sampling2d_1/ShapeShape3functional_19/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2%
#functional_19/up_sampling2d_1/Shape°
1functional_19/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1functional_19/up_sampling2d_1/strided_slice/stack´
3functional_19/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_1/strided_slice/stack_1´
3functional_19/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_1/strided_slice/stack_2
+functional_19/up_sampling2d_1/strided_sliceStridedSlice,functional_19/up_sampling2d_1/Shape:output:0:functional_19/up_sampling2d_1/strided_slice/stack:output:0<functional_19/up_sampling2d_1/strided_slice/stack_1:output:0<functional_19/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+functional_19/up_sampling2d_1/strided_slice
#functional_19/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#functional_19/up_sampling2d_1/ConstÖ
!functional_19/up_sampling2d_1/mulMul4functional_19/up_sampling2d_1/strided_slice:output:0,functional_19/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2#
!functional_19/up_sampling2d_1/mulÂ
:functional_19/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor3functional_19/leaky_re_lu_3/LeakyRelu:activations:0%functional_19/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2<
:functional_19/up_sampling2d_1/resize/ResizeNearestNeighborÜ
,functional_19/conv2d_6/Conv2D/ReadVariableOpReadVariableOp5functional_19_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,functional_19/conv2d_6/Conv2D/ReadVariableOpŽ
functional_19/conv2d_6/Conv2DConv2DKfunctional_19/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:04functional_19/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
functional_19/conv2d_6/Conv2DŇ
-functional_19/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp6functional_19_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-functional_19/conv2d_6/BiasAdd/ReadVariableOpĺ
functional_19/conv2d_6/BiasAddBiasAdd&functional_19/conv2d_6/Conv2D:output:05functional_19/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/conv2d_6/BiasAdd
 functional_19/pixel_norm_4/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_4/pow/yÖ
functional_19/pixel_norm_4/powPow'functional_19/conv2d_6/BiasAdd:output:0)functional_19/pixel_norm_4/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_4/pową
1functional_19/pixel_norm_4/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_4/Mean/reduction_indiceső
functional_19/pixel_norm_4/MeanMean"functional_19/pixel_norm_4/pow:z:0:functional_19/pixel_norm_4/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2!
functional_19/pixel_norm_4/Mean
 functional_19/pixel_norm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_4/add/yŘ
functional_19/pixel_norm_4/addAddV2(functional_19/pixel_norm_4/Mean:output:0)functional_19/pixel_norm_4/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_4/add
 functional_19/pixel_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_4/Const
"functional_19/pixel_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_4/Const_1ú
0functional_19/pixel_norm_4/clip_by_value/MinimumMinimum"functional_19/pixel_norm_4/add:z:0+functional_19/pixel_norm_4/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙22
0functional_19/pixel_norm_4/clip_by_value/Minimumú
(functional_19/pixel_norm_4/clip_by_valueMaximum4functional_19/pixel_norm_4/clip_by_value/Minimum:z:0)functional_19/pixel_norm_4/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(functional_19/pixel_norm_4/clip_by_value˛
functional_19/pixel_norm_4/SqrtSqrt,functional_19/pixel_norm_4/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_4/SqrtÜ
"functional_19/pixel_norm_4/truedivRealDiv'functional_19/conv2d_6/BiasAdd:output:0#functional_19/pixel_norm_4/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"functional_19/pixel_norm_4/truedivľ
%functional_19/leaky_re_lu_4/LeakyRelu	LeakyRelu&functional_19/pixel_norm_4/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%functional_19/leaky_re_lu_4/LeakyReluÜ
,functional_19/conv2d_7/Conv2D/ReadVariableOpReadVariableOp5functional_19_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,functional_19/conv2d_7/Conv2D/ReadVariableOp
functional_19/conv2d_7/Conv2DConv2D3functional_19/leaky_re_lu_4/LeakyRelu:activations:04functional_19/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
functional_19/conv2d_7/Conv2DŇ
-functional_19/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp6functional_19_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-functional_19/conv2d_7/BiasAdd/ReadVariableOpĺ
functional_19/conv2d_7/BiasAddBiasAdd&functional_19/conv2d_7/Conv2D:output:05functional_19/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/conv2d_7/BiasAdd
 functional_19/pixel_norm_5/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_5/pow/yÖ
functional_19/pixel_norm_5/powPow'functional_19/conv2d_7/BiasAdd:output:0)functional_19/pixel_norm_5/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_5/pową
1functional_19/pixel_norm_5/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_5/Mean/reduction_indiceső
functional_19/pixel_norm_5/MeanMean"functional_19/pixel_norm_5/pow:z:0:functional_19/pixel_norm_5/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2!
functional_19/pixel_norm_5/Mean
 functional_19/pixel_norm_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_5/add/yŘ
functional_19/pixel_norm_5/addAddV2(functional_19/pixel_norm_5/Mean:output:0)functional_19/pixel_norm_5/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
functional_19/pixel_norm_5/add
 functional_19/pixel_norm_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_5/Const
"functional_19/pixel_norm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_5/Const_1ú
0functional_19/pixel_norm_5/clip_by_value/MinimumMinimum"functional_19/pixel_norm_5/add:z:0+functional_19/pixel_norm_5/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙22
0functional_19/pixel_norm_5/clip_by_value/Minimumú
(functional_19/pixel_norm_5/clip_by_valueMaximum4functional_19/pixel_norm_5/clip_by_value/Minimum:z:0)functional_19/pixel_norm_5/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(functional_19/pixel_norm_5/clip_by_value˛
functional_19/pixel_norm_5/SqrtSqrt,functional_19/pixel_norm_5/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_5/SqrtÜ
"functional_19/pixel_norm_5/truedivRealDiv'functional_19/conv2d_7/BiasAdd:output:0#functional_19/pixel_norm_5/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"functional_19/pixel_norm_5/truedivľ
%functional_19/leaky_re_lu_5/LeakyRelu	LeakyRelu&functional_19/pixel_norm_5/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%functional_19/leaky_re_lu_5/LeakyRelu­
#functional_19/up_sampling2d_2/ShapeShape3functional_19/leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2%
#functional_19/up_sampling2d_2/Shape°
1functional_19/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1functional_19/up_sampling2d_2/strided_slice/stack´
3functional_19/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_2/strided_slice/stack_1´
3functional_19/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_2/strided_slice/stack_2
+functional_19/up_sampling2d_2/strided_sliceStridedSlice,functional_19/up_sampling2d_2/Shape:output:0:functional_19/up_sampling2d_2/strided_slice/stack:output:0<functional_19/up_sampling2d_2/strided_slice/stack_1:output:0<functional_19/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+functional_19/up_sampling2d_2/strided_slice
#functional_19/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#functional_19/up_sampling2d_2/ConstÖ
!functional_19/up_sampling2d_2/mulMul4functional_19/up_sampling2d_2/strided_slice:output:0,functional_19/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2#
!functional_19/up_sampling2d_2/mulÂ
:functional_19/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor3functional_19/leaky_re_lu_5/LeakyRelu:activations:0%functional_19/up_sampling2d_2/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
half_pixel_centers(2<
:functional_19/up_sampling2d_2/resize/ResizeNearestNeighborÜ
,functional_19/conv2d_9/Conv2D/ReadVariableOpReadVariableOp5functional_19_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02.
,functional_19/conv2d_9/Conv2D/ReadVariableOpŽ
functional_19/conv2d_9/Conv2DConv2DKfunctional_19/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:04functional_19/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
paddingSAME*
strides
2
functional_19/conv2d_9/Conv2DŇ
-functional_19/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp6functional_19_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-functional_19/conv2d_9/BiasAdd/ReadVariableOpĺ
functional_19/conv2d_9/BiasAddBiasAdd&functional_19/conv2d_9/Conv2D:output:05functional_19/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2 
functional_19/conv2d_9/BiasAdd
 functional_19/pixel_norm_6/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_6/pow/yÖ
functional_19/pixel_norm_6/powPow'functional_19/conv2d_9/BiasAdd:output:0)functional_19/pixel_norm_6/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2 
functional_19/pixel_norm_6/pową
1functional_19/pixel_norm_6/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_6/Mean/reduction_indiceső
functional_19/pixel_norm_6/MeanMean"functional_19/pixel_norm_6/pow:z:0:functional_19/pixel_norm_6/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
	keep_dims(2!
functional_19/pixel_norm_6/Mean
 functional_19/pixel_norm_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_6/add/yŘ
functional_19/pixel_norm_6/addAddV2(functional_19/pixel_norm_6/Mean:output:0)functional_19/pixel_norm_6/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2 
functional_19/pixel_norm_6/add
 functional_19/pixel_norm_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_6/Const
"functional_19/pixel_norm_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_6/Const_1ú
0functional_19/pixel_norm_6/clip_by_value/MinimumMinimum"functional_19/pixel_norm_6/add:z:0+functional_19/pixel_norm_6/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  22
0functional_19/pixel_norm_6/clip_by_value/Minimumú
(functional_19/pixel_norm_6/clip_by_valueMaximum4functional_19/pixel_norm_6/clip_by_value/Minimum:z:0)functional_19/pixel_norm_6/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2*
(functional_19/pixel_norm_6/clip_by_value˛
functional_19/pixel_norm_6/SqrtSqrt,functional_19/pixel_norm_6/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2!
functional_19/pixel_norm_6/SqrtÜ
"functional_19/pixel_norm_6/truedivRealDiv'functional_19/conv2d_9/BiasAdd:output:0#functional_19/pixel_norm_6/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2$
"functional_19/pixel_norm_6/truedivľ
%functional_19/leaky_re_lu_6/LeakyRelu	LeakyRelu&functional_19/pixel_norm_6/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2'
%functional_19/leaky_re_lu_6/LeakyReluß
-functional_19/conv2d_10/Conv2D/ReadVariableOpReadVariableOp6functional_19_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02/
-functional_19/conv2d_10/Conv2D/ReadVariableOp
functional_19/conv2d_10/Conv2DConv2D3functional_19/leaky_re_lu_6/LeakyRelu:activations:05functional_19/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
paddingSAME*
strides
2 
functional_19/conv2d_10/Conv2DŐ
.functional_19/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp7functional_19_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_19/conv2d_10/BiasAdd/ReadVariableOpé
functional_19/conv2d_10/BiasAddBiasAdd'functional_19/conv2d_10/Conv2D:output:06functional_19/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2!
functional_19/conv2d_10/BiasAdd
 functional_19/pixel_norm_7/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_7/pow/y×
functional_19/pixel_norm_7/powPow(functional_19/conv2d_10/BiasAdd:output:0)functional_19/pixel_norm_7/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2 
functional_19/pixel_norm_7/pową
1functional_19/pixel_norm_7/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_7/Mean/reduction_indiceső
functional_19/pixel_norm_7/MeanMean"functional_19/pixel_norm_7/pow:z:0:functional_19/pixel_norm_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
	keep_dims(2!
functional_19/pixel_norm_7/Mean
 functional_19/pixel_norm_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_7/add/yŘ
functional_19/pixel_norm_7/addAddV2(functional_19/pixel_norm_7/Mean:output:0)functional_19/pixel_norm_7/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2 
functional_19/pixel_norm_7/add
 functional_19/pixel_norm_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_7/Const
"functional_19/pixel_norm_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_7/Const_1ú
0functional_19/pixel_norm_7/clip_by_value/MinimumMinimum"functional_19/pixel_norm_7/add:z:0+functional_19/pixel_norm_7/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  22
0functional_19/pixel_norm_7/clip_by_value/Minimumú
(functional_19/pixel_norm_7/clip_by_valueMaximum4functional_19/pixel_norm_7/clip_by_value/Minimum:z:0)functional_19/pixel_norm_7/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2*
(functional_19/pixel_norm_7/clip_by_value˛
functional_19/pixel_norm_7/SqrtSqrt,functional_19/pixel_norm_7/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2!
functional_19/pixel_norm_7/SqrtÝ
"functional_19/pixel_norm_7/truedivRealDiv(functional_19/conv2d_10/BiasAdd:output:0#functional_19/pixel_norm_7/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2$
"functional_19/pixel_norm_7/truedivľ
%functional_19/leaky_re_lu_7/LeakyRelu	LeakyRelu&functional_19/pixel_norm_7/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2'
%functional_19/leaky_re_lu_7/LeakyRelu­
#functional_19/up_sampling2d_3/ShapeShape3functional_19/leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2%
#functional_19/up_sampling2d_3/Shape°
1functional_19/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1functional_19/up_sampling2d_3/strided_slice/stack´
3functional_19/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_3/strided_slice/stack_1´
3functional_19/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_3/strided_slice/stack_2
+functional_19/up_sampling2d_3/strided_sliceStridedSlice,functional_19/up_sampling2d_3/Shape:output:0:functional_19/up_sampling2d_3/strided_slice/stack:output:0<functional_19/up_sampling2d_3/strided_slice/stack_1:output:0<functional_19/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+functional_19/up_sampling2d_3/strided_slice
#functional_19/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#functional_19/up_sampling2d_3/ConstÖ
!functional_19/up_sampling2d_3/mulMul4functional_19/up_sampling2d_3/strided_slice:output:0,functional_19/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2#
!functional_19/up_sampling2d_3/mulÂ
:functional_19/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor3functional_19/leaky_re_lu_7/LeakyRelu:activations:0%functional_19/up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
half_pixel_centers(2<
:functional_19/up_sampling2d_3/resize/ResizeNearestNeighborß
-functional_19/conv2d_12/Conv2D/ReadVariableOpReadVariableOp6functional_19_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02/
-functional_19/conv2d_12/Conv2D/ReadVariableOpą
functional_19/conv2d_12/Conv2DConv2DKfunctional_19/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:05functional_19/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
paddingSAME*
strides
2 
functional_19/conv2d_12/Conv2DŐ
.functional_19/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp7functional_19_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_19/conv2d_12/BiasAdd/ReadVariableOpé
functional_19/conv2d_12/BiasAddBiasAdd'functional_19/conv2d_12/Conv2D:output:06functional_19/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2!
functional_19/conv2d_12/BiasAdd
 functional_19/pixel_norm_8/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_8/pow/y×
functional_19/pixel_norm_8/powPow(functional_19/conv2d_12/BiasAdd:output:0)functional_19/pixel_norm_8/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2 
functional_19/pixel_norm_8/pową
1functional_19/pixel_norm_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_8/Mean/reduction_indiceső
functional_19/pixel_norm_8/MeanMean"functional_19/pixel_norm_8/pow:z:0:functional_19/pixel_norm_8/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
	keep_dims(2!
functional_19/pixel_norm_8/Mean
 functional_19/pixel_norm_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_8/add/yŘ
functional_19/pixel_norm_8/addAddV2(functional_19/pixel_norm_8/Mean:output:0)functional_19/pixel_norm_8/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2 
functional_19/pixel_norm_8/add
 functional_19/pixel_norm_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_8/Const
"functional_19/pixel_norm_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_8/Const_1ú
0functional_19/pixel_norm_8/clip_by_value/MinimumMinimum"functional_19/pixel_norm_8/add:z:0+functional_19/pixel_norm_8/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@22
0functional_19/pixel_norm_8/clip_by_value/Minimumú
(functional_19/pixel_norm_8/clip_by_valueMaximum4functional_19/pixel_norm_8/clip_by_value/Minimum:z:0)functional_19/pixel_norm_8/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2*
(functional_19/pixel_norm_8/clip_by_value˛
functional_19/pixel_norm_8/SqrtSqrt,functional_19/pixel_norm_8/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2!
functional_19/pixel_norm_8/SqrtÝ
"functional_19/pixel_norm_8/truedivRealDiv(functional_19/conv2d_12/BiasAdd:output:0#functional_19/pixel_norm_8/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2$
"functional_19/pixel_norm_8/truedivľ
%functional_19/leaky_re_lu_8/LeakyRelu	LeakyRelu&functional_19/pixel_norm_8/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2'
%functional_19/leaky_re_lu_8/LeakyReluß
-functional_19/conv2d_13/Conv2D/ReadVariableOpReadVariableOp6functional_19_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02/
-functional_19/conv2d_13/Conv2D/ReadVariableOp
functional_19/conv2d_13/Conv2DConv2D3functional_19/leaky_re_lu_8/LeakyRelu:activations:05functional_19/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
paddingSAME*
strides
2 
functional_19/conv2d_13/Conv2DŐ
.functional_19/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp7functional_19_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_19/conv2d_13/BiasAdd/ReadVariableOpé
functional_19/conv2d_13/BiasAddBiasAdd'functional_19/conv2d_13/Conv2D:output:06functional_19/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2!
functional_19/conv2d_13/BiasAdd
 functional_19/pixel_norm_9/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 functional_19/pixel_norm_9/pow/y×
functional_19/pixel_norm_9/powPow(functional_19/conv2d_13/BiasAdd:output:0)functional_19/pixel_norm_9/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2 
functional_19/pixel_norm_9/pową
1functional_19/pixel_norm_9/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_9/Mean/reduction_indiceső
functional_19/pixel_norm_9/MeanMean"functional_19/pixel_norm_9/pow:z:0:functional_19/pixel_norm_9/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
	keep_dims(2!
functional_19/pixel_norm_9/Mean
 functional_19/pixel_norm_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22"
 functional_19/pixel_norm_9/add/yŘ
functional_19/pixel_norm_9/addAddV2(functional_19/pixel_norm_9/Mean:output:0)functional_19/pixel_norm_9/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2 
functional_19/pixel_norm_9/add
 functional_19/pixel_norm_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 functional_19/pixel_norm_9/Const
"functional_19/pixel_norm_9/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2$
"functional_19/pixel_norm_9/Const_1ú
0functional_19/pixel_norm_9/clip_by_value/MinimumMinimum"functional_19/pixel_norm_9/add:z:0+functional_19/pixel_norm_9/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@22
0functional_19/pixel_norm_9/clip_by_value/Minimumú
(functional_19/pixel_norm_9/clip_by_valueMaximum4functional_19/pixel_norm_9/clip_by_value/Minimum:z:0)functional_19/pixel_norm_9/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2*
(functional_19/pixel_norm_9/clip_by_value˛
functional_19/pixel_norm_9/SqrtSqrt,functional_19/pixel_norm_9/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2!
functional_19/pixel_norm_9/SqrtÝ
"functional_19/pixel_norm_9/truedivRealDiv(functional_19/conv2d_13/BiasAdd:output:0#functional_19/pixel_norm_9/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2$
"functional_19/pixel_norm_9/truedivľ
%functional_19/leaky_re_lu_9/LeakyRelu	LeakyRelu&functional_19/pixel_norm_9/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2'
%functional_19/leaky_re_lu_9/LeakyRelu­
#functional_19/up_sampling2d_4/ShapeShape3functional_19/leaky_re_lu_9/LeakyRelu:activations:0*
T0*
_output_shapes
:2%
#functional_19/up_sampling2d_4/Shape°
1functional_19/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1functional_19/up_sampling2d_4/strided_slice/stack´
3functional_19/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_4/strided_slice/stack_1´
3functional_19/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_19/up_sampling2d_4/strided_slice/stack_2
+functional_19/up_sampling2d_4/strided_sliceStridedSlice,functional_19/up_sampling2d_4/Shape:output:0:functional_19/up_sampling2d_4/strided_slice/stack:output:0<functional_19/up_sampling2d_4/strided_slice/stack_1:output:0<functional_19/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+functional_19/up_sampling2d_4/strided_slice
#functional_19/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#functional_19/up_sampling2d_4/ConstÖ
!functional_19/up_sampling2d_4/mulMul4functional_19/up_sampling2d_4/strided_slice:output:0,functional_19/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2#
!functional_19/up_sampling2d_4/mulÄ
:functional_19/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor3functional_19/leaky_re_lu_9/LeakyRelu:activations:0%functional_19/up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2<
:functional_19/up_sampling2d_4/resize/ResizeNearestNeighborß
-functional_19/conv2d_15/Conv2D/ReadVariableOpReadVariableOp6functional_19_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02/
-functional_19/conv2d_15/Conv2D/ReadVariableOpł
functional_19/conv2d_15/Conv2DConv2DKfunctional_19/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:05functional_19/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2 
functional_19/conv2d_15/Conv2DŐ
.functional_19/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp7functional_19_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_19/conv2d_15/BiasAdd/ReadVariableOpë
functional_19/conv2d_15/BiasAddBiasAdd'functional_19/conv2d_15/Conv2D:output:06functional_19/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2!
functional_19/conv2d_15/BiasAdd
!functional_19/pixel_norm_10/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2#
!functional_19/pixel_norm_10/pow/yÜ
functional_19/pixel_norm_10/powPow(functional_19/conv2d_15/BiasAdd:output:0*functional_19/pixel_norm_10/pow/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_10/powł
2functional_19/pixel_norm_10/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙24
2functional_19/pixel_norm_10/Mean/reduction_indicesű
 functional_19/pixel_norm_10/MeanMean#functional_19/pixel_norm_10/pow:z:0;functional_19/pixel_norm_10/Mean/reduction_indices:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2"
 functional_19/pixel_norm_10/Mean
!functional_19/pixel_norm_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22#
!functional_19/pixel_norm_10/add/yŢ
functional_19/pixel_norm_10/addAddV2)functional_19/pixel_norm_10/Mean:output:0*functional_19/pixel_norm_10/add/y:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_10/add
!functional_19/pixel_norm_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!functional_19/pixel_norm_10/Const
#functional_19/pixel_norm_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2%
#functional_19/pixel_norm_10/Const_1
1functional_19/pixel_norm_10/clip_by_value/MinimumMinimum#functional_19/pixel_norm_10/add:z:0,functional_19/pixel_norm_10/Const_1:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_10/clip_by_value/Minimum
)functional_19/pixel_norm_10/clip_by_valueMaximum5functional_19/pixel_norm_10/clip_by_value/Minimum:z:0*functional_19/pixel_norm_10/Const:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)functional_19/pixel_norm_10/clip_by_valueˇ
 functional_19/pixel_norm_10/SqrtSqrt-functional_19/pixel_norm_10/clip_by_value:z:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 functional_19/pixel_norm_10/Sqrtâ
#functional_19/pixel_norm_10/truedivRealDiv(functional_19/conv2d_15/BiasAdd:output:0$functional_19/pixel_norm_10/Sqrt:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2%
#functional_19/pixel_norm_10/truedivş
&functional_19/leaky_re_lu_10/LeakyRelu	LeakyRelu'functional_19/pixel_norm_10/truediv:z:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2(
&functional_19/leaky_re_lu_10/LeakyReluß
-functional_19/conv2d_16/Conv2D/ReadVariableOpReadVariableOp6functional_19_conv2d_16_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02/
-functional_19/conv2d_16/Conv2D/ReadVariableOp
functional_19/conv2d_16/Conv2DConv2D4functional_19/leaky_re_lu_10/LeakyRelu:activations:05functional_19/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2 
functional_19/conv2d_16/Conv2DŐ
.functional_19/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp7functional_19_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_19/conv2d_16/BiasAdd/ReadVariableOpë
functional_19/conv2d_16/BiasAddBiasAdd'functional_19/conv2d_16/Conv2D:output:06functional_19/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2!
functional_19/conv2d_16/BiasAdd
!functional_19/pixel_norm_11/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2#
!functional_19/pixel_norm_11/pow/yÜ
functional_19/pixel_norm_11/powPow(functional_19/conv2d_16/BiasAdd:output:0*functional_19/pixel_norm_11/pow/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_11/powł
2functional_19/pixel_norm_11/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙24
2functional_19/pixel_norm_11/Mean/reduction_indicesű
 functional_19/pixel_norm_11/MeanMean#functional_19/pixel_norm_11/pow:z:0;functional_19/pixel_norm_11/Mean/reduction_indices:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2"
 functional_19/pixel_norm_11/Mean
!functional_19/pixel_norm_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22#
!functional_19/pixel_norm_11/add/yŢ
functional_19/pixel_norm_11/addAddV2)functional_19/pixel_norm_11/Mean:output:0*functional_19/pixel_norm_11/add/y:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/pixel_norm_11/add
!functional_19/pixel_norm_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!functional_19/pixel_norm_11/Const
#functional_19/pixel_norm_11/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2%
#functional_19/pixel_norm_11/Const_1
1functional_19/pixel_norm_11/clip_by_value/MinimumMinimum#functional_19/pixel_norm_11/add:z:0,functional_19/pixel_norm_11/Const_1:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙23
1functional_19/pixel_norm_11/clip_by_value/Minimum
)functional_19/pixel_norm_11/clip_by_valueMaximum5functional_19/pixel_norm_11/clip_by_value/Minimum:z:0*functional_19/pixel_norm_11/Const:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2+
)functional_19/pixel_norm_11/clip_by_valueˇ
 functional_19/pixel_norm_11/SqrtSqrt-functional_19/pixel_norm_11/clip_by_value:z:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 functional_19/pixel_norm_11/Sqrtâ
#functional_19/pixel_norm_11/truedivRealDiv(functional_19/conv2d_16/BiasAdd:output:0$functional_19/pixel_norm_11/Sqrt:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2%
#functional_19/pixel_norm_11/truedivş
&functional_19/leaky_re_lu_11/LeakyRelu	LeakyRelu'functional_19/pixel_norm_11/truediv:z:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2(
&functional_19/leaky_re_lu_11/LeakyReluŢ
-functional_19/conv2d_17/Conv2D/ReadVariableOpReadVariableOp6functional_19_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02/
-functional_19/conv2d_17/Conv2D/ReadVariableOp
functional_19/conv2d_17/Conv2DConv2D4functional_19/leaky_re_lu_11/LeakyRelu:activations:05functional_19/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2 
functional_19/conv2d_17/Conv2DÔ
.functional_19/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp7functional_19_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_19/conv2d_17/BiasAdd/ReadVariableOpę
functional_19/conv2d_17/BiasAddBiasAdd'functional_19/conv2d_17/Conv2D:output:06functional_19/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
functional_19/conv2d_17/BiasAddŞ
functional_19/conv2d_17/TanhTanh(functional_19/conv2d_17/BiasAdd:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_19/conv2d_17/Tanh~
IdentityIdentity functional_19/conv2d_17/Tanh:y:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b:::::::::::::::::::::::::::::P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
!
_user_specified_name	input_1

H
.__inference_pixel_norm_9_layer_call_fn_6188551
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_61866372
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Í
c
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_6188296
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
°
f
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_6188556

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


*__inference_conv2d_1_layer_call_fn_6188130

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_61860972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

K
/__inference_leaky_re_lu_6_layer_call_fn_6188411

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_61864602
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ

+__inference_conv2d_10_layer_call_fn_6188430

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_61864782
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
š
F
,__inference_pixel_norm_layer_call_fn_6188101
data
identityĎ
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_pixel_norm_layer_call_and_return_conditional_losses_61860662
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:V R
0
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namedata
ĺ
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6188106

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

K
/__inference_leaky_re_lu_2_layer_call_fn_6188211

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_61862062
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
d
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_6188596
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
˛
M
1__inference_up_sampling2d_4_layer_call_fn_6185972

inputs
identityđ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_61859662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

H
.__inference_pixel_norm_3_layer_call_fn_6188251
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_61862562
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

H
.__inference_pixel_norm_5_layer_call_fn_6188351
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_61863832
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Í
c
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_6188396
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

}
(__inference_conv2d_layer_call_fn_6188080

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_61860342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ž
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6188421

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_6188456

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

H
.__inference_pixel_norm_4_layer_call_fn_6188301
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_61863202
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

h
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6185947

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulŐ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛
M
1__inference_up_sampling2d_1_layer_call_fn_6185915

inputs
identityđ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_61859092
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
d
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_6188646
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
	
­
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6188371

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
Ş
B__inference_dense_layer_call_and_return_conditional_losses_6185986

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	b*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙b:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_6186460

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛
M
1__inference_up_sampling2d_3_layer_call_fn_6185953

inputs
identityđ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_61859472
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ž
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6188471

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ś

%__inference_signature_wrapper_6187311
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity˘StatefulPartitionedCallĂ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_61858772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
!
_user_specified_name	input_1

H
.__inference_pixel_norm_8_layer_call_fn_6188501
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_61865742
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

K
/__inference_leaky_re_lu_9_layer_call_fn_6188561

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_61866502
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6188171

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

H
.__inference_pixel_norm_7_layer_call_fn_6188451
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_61865102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

K
/__inference_leaky_re_lu_7_layer_call_fn_6188461

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_61865232
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ţ
ĺ

J__inference_functional_19_layer_call_and_return_conditional_losses_6187606

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource
identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	b*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/BiasAddd
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape/Reshape/shape/3ę
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape 
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
reshape/ReshapeŹ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpË
conv2d/Conv2DConv2Dreshape/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d/Conv2D˘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2d/BiasAdd/ReadVariableOpĽ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d/BiasAddi
pixel_norm/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm/pow/y
pixel_norm/powPowconv2d/BiasAdd:output:0pixel_norm/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/pow
!pixel_norm/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2#
!pixel_norm/Mean/reduction_indicesľ
pixel_norm/MeanMeanpixel_norm/pow:z:0*pixel_norm/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm/Meani
pixel_norm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm/add/y
pixel_norm/addAddV2pixel_norm/Mean:output:0pixel_norm/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/addi
pixel_norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm/Constm
pixel_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm/Const_1ş
 pixel_norm/clip_by_value/MinimumMinimumpixel_norm/add:z:0pixel_norm/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 pixel_norm/clip_by_value/Minimumş
pixel_norm/clip_by_valueMaximum$pixel_norm/clip_by_value/Minimum:z:0pixel_norm/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/clip_by_value
pixel_norm/SqrtSqrtpixel_norm/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/Sqrt
pixel_norm/truedivRealDivconv2d/BiasAdd:output:0pixel_norm/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/truediv
leaky_re_lu/LeakyRelu	LeakyRelupixel_norm/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu/LeakyRelu˛
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÜ
conv2d_1/Conv2DConv2D#leaky_re_lu/LeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp­
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_1/BiasAddm
pixel_norm_1/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_1/pow/y
pixel_norm_1/powPowconv2d_1/BiasAdd:output:0pixel_norm_1/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/pow
#pixel_norm_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_1/Mean/reduction_indices˝
pixel_norm_1/MeanMeanpixel_norm_1/pow:z:0,pixel_norm_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_1/Meanm
pixel_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_1/add/y 
pixel_norm_1/addAddV2pixel_norm_1/Mean:output:0pixel_norm_1/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/addm
pixel_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_1/Constq
pixel_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_1/Const_1Â
"pixel_norm_1/clip_by_value/MinimumMinimumpixel_norm_1/add:z:0pixel_norm_1/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_1/clip_by_value/MinimumÂ
pixel_norm_1/clip_by_valueMaximum&pixel_norm_1/clip_by_value/Minimum:z:0pixel_norm_1/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/clip_by_value
pixel_norm_1/SqrtSqrtpixel_norm_1/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/Sqrt¤
pixel_norm_1/truedivRealDivconv2d_1/BiasAdd:output:0pixel_norm_1/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/truediv
leaky_re_lu_1/LeakyRelu	LeakyRelupixel_norm_1/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_1/LeakyRelu
up_sampling2d/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2˘
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_1/LeakyRelu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor˛
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpô
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_3/Conv2D¨
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_3/BiasAddm
pixel_norm_2/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_2/pow/y
pixel_norm_2/powPowconv2d_3/BiasAdd:output:0pixel_norm_2/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/pow
#pixel_norm_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_2/Mean/reduction_indices˝
pixel_norm_2/MeanMeanpixel_norm_2/pow:z:0,pixel_norm_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_2/Meanm
pixel_norm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_2/add/y 
pixel_norm_2/addAddV2pixel_norm_2/Mean:output:0pixel_norm_2/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/addm
pixel_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_2/Constq
pixel_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_2/Const_1Â
"pixel_norm_2/clip_by_value/MinimumMinimumpixel_norm_2/add:z:0pixel_norm_2/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_2/clip_by_value/MinimumÂ
pixel_norm_2/clip_by_valueMaximum&pixel_norm_2/clip_by_value/Minimum:z:0pixel_norm_2/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/clip_by_value
pixel_norm_2/SqrtSqrtpixel_norm_2/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/Sqrt¤
pixel_norm_2/truedivRealDivconv2d_3/BiasAdd:output:0pixel_norm_2/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/truediv
leaky_re_lu_2/LeakyRelu	LeakyRelupixel_norm_2/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_2/LeakyRelu˛
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpŢ
conv2d_4/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_4/BiasAddm
pixel_norm_3/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_3/pow/y
pixel_norm_3/powPowconv2d_4/BiasAdd:output:0pixel_norm_3/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/pow
#pixel_norm_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_3/Mean/reduction_indices˝
pixel_norm_3/MeanMeanpixel_norm_3/pow:z:0,pixel_norm_3/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_3/Meanm
pixel_norm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_3/add/y 
pixel_norm_3/addAddV2pixel_norm_3/Mean:output:0pixel_norm_3/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/addm
pixel_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_3/Constq
pixel_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_3/Const_1Â
"pixel_norm_3/clip_by_value/MinimumMinimumpixel_norm_3/add:z:0pixel_norm_3/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_3/clip_by_value/MinimumÂ
pixel_norm_3/clip_by_valueMaximum&pixel_norm_3/clip_by_value/Minimum:z:0pixel_norm_3/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/clip_by_value
pixel_norm_3/SqrtSqrtpixel_norm_3/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/Sqrt¤
pixel_norm_3/truedivRealDivconv2d_4/BiasAdd:output:0pixel_norm_3/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/truediv
leaky_re_lu_3/LeakyRelu	LeakyRelupixel_norm_3/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_3/LeakyRelu
up_sampling2d_1/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2Ž
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_3/LeakyRelu:activations:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor˛
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOpö
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_6/BiasAddm
pixel_norm_4/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_4/pow/y
pixel_norm_4/powPowconv2d_6/BiasAdd:output:0pixel_norm_4/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/pow
#pixel_norm_4/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_4/Mean/reduction_indices˝
pixel_norm_4/MeanMeanpixel_norm_4/pow:z:0,pixel_norm_4/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_4/Meanm
pixel_norm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_4/add/y 
pixel_norm_4/addAddV2pixel_norm_4/Mean:output:0pixel_norm_4/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/addm
pixel_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_4/Constq
pixel_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_4/Const_1Â
"pixel_norm_4/clip_by_value/MinimumMinimumpixel_norm_4/add:z:0pixel_norm_4/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_4/clip_by_value/MinimumÂ
pixel_norm_4/clip_by_valueMaximum&pixel_norm_4/clip_by_value/Minimum:z:0pixel_norm_4/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/clip_by_value
pixel_norm_4/SqrtSqrtpixel_norm_4/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/Sqrt¤
pixel_norm_4/truedivRealDivconv2d_6/BiasAdd:output:0pixel_norm_4/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/truediv
leaky_re_lu_4/LeakyRelu	LeakyRelupixel_norm_4/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_4/LeakyRelu˛
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOpŢ
conv2d_7/Conv2DConv2D%leaky_re_lu_4/LeakyRelu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_7/Conv2D¨
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp­
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_7/BiasAddm
pixel_norm_5/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_5/pow/y
pixel_norm_5/powPowconv2d_7/BiasAdd:output:0pixel_norm_5/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/pow
#pixel_norm_5/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_5/Mean/reduction_indices˝
pixel_norm_5/MeanMeanpixel_norm_5/pow:z:0,pixel_norm_5/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_5/Meanm
pixel_norm_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_5/add/y 
pixel_norm_5/addAddV2pixel_norm_5/Mean:output:0pixel_norm_5/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/addm
pixel_norm_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_5/Constq
pixel_norm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_5/Const_1Â
"pixel_norm_5/clip_by_value/MinimumMinimumpixel_norm_5/add:z:0pixel_norm_5/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_5/clip_by_value/MinimumÂ
pixel_norm_5/clip_by_valueMaximum&pixel_norm_5/clip_by_value/Minimum:z:0pixel_norm_5/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/clip_by_value
pixel_norm_5/SqrtSqrtpixel_norm_5/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/Sqrt¤
pixel_norm_5/truedivRealDivconv2d_7/BiasAdd:output:0pixel_norm_5/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/truediv
leaky_re_lu_5/LeakyRelu	LeakyRelupixel_norm_5/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_5/LeakyRelu
up_sampling2d_2/ShapeShape%leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2Ž
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_5/LeakyRelu:activations:0up_sampling2d_2/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor˛
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOpö
conv2d_9/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
paddingSAME*
strides
2
conv2d_9/Conv2D¨
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp­
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
conv2d_9/BiasAddm
pixel_norm_6/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_6/pow/y
pixel_norm_6/powPowconv2d_9/BiasAdd:output:0pixel_norm_6/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/pow
#pixel_norm_6/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_6/Mean/reduction_indices˝
pixel_norm_6/MeanMeanpixel_norm_6/pow:z:0,pixel_norm_6/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
	keep_dims(2
pixel_norm_6/Meanm
pixel_norm_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_6/add/y 
pixel_norm_6/addAddV2pixel_norm_6/Mean:output:0pixel_norm_6/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/addm
pixel_norm_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_6/Constq
pixel_norm_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_6/Const_1Â
"pixel_norm_6/clip_by_value/MinimumMinimumpixel_norm_6/add:z:0pixel_norm_6/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2$
"pixel_norm_6/clip_by_value/MinimumÂ
pixel_norm_6/clip_by_valueMaximum&pixel_norm_6/clip_by_value/Minimum:z:0pixel_norm_6/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/clip_by_value
pixel_norm_6/SqrtSqrtpixel_norm_6/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/Sqrt¤
pixel_norm_6/truedivRealDivconv2d_9/BiasAdd:output:0pixel_norm_6/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/truediv
leaky_re_lu_6/LeakyRelu	LeakyRelupixel_norm_6/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
leaky_re_lu_6/LeakyReluľ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOpá
conv2d_10/Conv2DConv2D%leaky_re_lu_6/LeakyRelu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
paddingSAME*
strides
2
conv2d_10/Conv2DŤ
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOpą
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
conv2d_10/BiasAddm
pixel_norm_7/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_7/pow/y
pixel_norm_7/powPowconv2d_10/BiasAdd:output:0pixel_norm_7/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/pow
#pixel_norm_7/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_7/Mean/reduction_indices˝
pixel_norm_7/MeanMeanpixel_norm_7/pow:z:0,pixel_norm_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
	keep_dims(2
pixel_norm_7/Meanm
pixel_norm_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_7/add/y 
pixel_norm_7/addAddV2pixel_norm_7/Mean:output:0pixel_norm_7/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/addm
pixel_norm_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_7/Constq
pixel_norm_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_7/Const_1Â
"pixel_norm_7/clip_by_value/MinimumMinimumpixel_norm_7/add:z:0pixel_norm_7/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2$
"pixel_norm_7/clip_by_value/MinimumÂ
pixel_norm_7/clip_by_valueMaximum&pixel_norm_7/clip_by_value/Minimum:z:0pixel_norm_7/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/clip_by_value
pixel_norm_7/SqrtSqrtpixel_norm_7/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/SqrtĽ
pixel_norm_7/truedivRealDivconv2d_10/BiasAdd:output:0pixel_norm_7/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/truediv
leaky_re_lu_7/LeakyRelu	LeakyRelupixel_norm_7/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
leaky_re_lu_7/LeakyRelu
up_sampling2d_3/ShapeShape%leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2Ž
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_7/LeakyRelu:activations:0up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighborľ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOpů
conv2d_12/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
paddingSAME*
strides
2
conv2d_12/Conv2DŤ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOpą
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
conv2d_12/BiasAddm
pixel_norm_8/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_8/pow/y
pixel_norm_8/powPowconv2d_12/BiasAdd:output:0pixel_norm_8/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/pow
#pixel_norm_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_8/Mean/reduction_indices˝
pixel_norm_8/MeanMeanpixel_norm_8/pow:z:0,pixel_norm_8/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
	keep_dims(2
pixel_norm_8/Meanm
pixel_norm_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_8/add/y 
pixel_norm_8/addAddV2pixel_norm_8/Mean:output:0pixel_norm_8/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/addm
pixel_norm_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_8/Constq
pixel_norm_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_8/Const_1Â
"pixel_norm_8/clip_by_value/MinimumMinimumpixel_norm_8/add:z:0pixel_norm_8/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2$
"pixel_norm_8/clip_by_value/MinimumÂ
pixel_norm_8/clip_by_valueMaximum&pixel_norm_8/clip_by_value/Minimum:z:0pixel_norm_8/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/clip_by_value
pixel_norm_8/SqrtSqrtpixel_norm_8/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/SqrtĽ
pixel_norm_8/truedivRealDivconv2d_12/BiasAdd:output:0pixel_norm_8/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/truediv
leaky_re_lu_8/LeakyRelu	LeakyRelupixel_norm_8/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
leaky_re_lu_8/LeakyReluľ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOpá
conv2d_13/Conv2DConv2D%leaky_re_lu_8/LeakyRelu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
paddingSAME*
strides
2
conv2d_13/Conv2DŤ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOpą
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
conv2d_13/BiasAddm
pixel_norm_9/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_9/pow/y
pixel_norm_9/powPowconv2d_13/BiasAdd:output:0pixel_norm_9/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/pow
#pixel_norm_9/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_9/Mean/reduction_indices˝
pixel_norm_9/MeanMeanpixel_norm_9/pow:z:0,pixel_norm_9/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
	keep_dims(2
pixel_norm_9/Meanm
pixel_norm_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_9/add/y 
pixel_norm_9/addAddV2pixel_norm_9/Mean:output:0pixel_norm_9/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/addm
pixel_norm_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_9/Constq
pixel_norm_9/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_9/Const_1Â
"pixel_norm_9/clip_by_value/MinimumMinimumpixel_norm_9/add:z:0pixel_norm_9/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2$
"pixel_norm_9/clip_by_value/MinimumÂ
pixel_norm_9/clip_by_valueMaximum&pixel_norm_9/clip_by_value/Minimum:z:0pixel_norm_9/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/clip_by_value
pixel_norm_9/SqrtSqrtpixel_norm_9/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/SqrtĽ
pixel_norm_9/truedivRealDivconv2d_13/BiasAdd:output:0pixel_norm_9/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/truediv
leaky_re_lu_9/LeakyRelu	LeakyRelupixel_norm_9/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
leaky_re_lu_9/LeakyRelu
up_sampling2d_4/ShapeShape%leaky_re_lu_9/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2Ž
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_9/LeakyRelu:activations:0up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighborľ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOpű
conv2d_15/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_15/Conv2DŤ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpł
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
conv2d_15/BiasAddo
pixel_norm_10/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_10/pow/y¤
pixel_norm_10/powPowconv2d_15/BiasAdd:output:0pixel_norm_10/pow/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/pow
$pixel_norm_10/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2&
$pixel_norm_10/Mean/reduction_indicesĂ
pixel_norm_10/MeanMeanpixel_norm_10/pow:z:0-pixel_norm_10/Mean/reduction_indices:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_10/Meano
pixel_norm_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_10/add/yŚ
pixel_norm_10/addAddV2pixel_norm_10/Mean:output:0pixel_norm_10/add/y:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/addo
pixel_norm_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_10/Consts
pixel_norm_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_10/Const_1Č
#pixel_norm_10/clip_by_value/MinimumMinimumpixel_norm_10/add:z:0pixel_norm_10/Const_1:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#pixel_norm_10/clip_by_value/MinimumČ
pixel_norm_10/clip_by_valueMaximum'pixel_norm_10/clip_by_value/Minimum:z:0pixel_norm_10/Const:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/clip_by_value
pixel_norm_10/SqrtSqrtpixel_norm_10/clip_by_value:z:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/SqrtŞ
pixel_norm_10/truedivRealDivconv2d_15/BiasAdd:output:0pixel_norm_10/Sqrt:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/truediv
leaky_re_lu_10/LeakyRelu	LeakyRelupixel_norm_10/truediv:z:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_10/LeakyReluľ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_16/Conv2D/ReadVariableOpä
conv2d_16/Conv2DConv2D&leaky_re_lu_10/LeakyRelu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_16/Conv2DŤ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOpł
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
conv2d_16/BiasAddo
pixel_norm_11/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_11/pow/y¤
pixel_norm_11/powPowconv2d_16/BiasAdd:output:0pixel_norm_11/pow/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/pow
$pixel_norm_11/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2&
$pixel_norm_11/Mean/reduction_indicesĂ
pixel_norm_11/MeanMeanpixel_norm_11/pow:z:0-pixel_norm_11/Mean/reduction_indices:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_11/Meano
pixel_norm_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_11/add/yŚ
pixel_norm_11/addAddV2pixel_norm_11/Mean:output:0pixel_norm_11/add/y:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/addo
pixel_norm_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_11/Consts
pixel_norm_11/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_11/Const_1Č
#pixel_norm_11/clip_by_value/MinimumMinimumpixel_norm_11/add:z:0pixel_norm_11/Const_1:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#pixel_norm_11/clip_by_value/MinimumČ
pixel_norm_11/clip_by_valueMaximum'pixel_norm_11/clip_by_value/Minimum:z:0pixel_norm_11/Const:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/clip_by_value
pixel_norm_11/SqrtSqrtpixel_norm_11/clip_by_value:z:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/SqrtŞ
pixel_norm_11/truedivRealDivconv2d_16/BiasAdd:output:0pixel_norm_11/Sqrt:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/truediv
leaky_re_lu_11/LeakyRelu	LeakyRelupixel_norm_11/truediv:z:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_11/LeakyRelu´
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_17/Conv2D/ReadVariableOpă
conv2d_17/Conv2DConv2D&leaky_re_lu_11/LeakyRelu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_17/Conv2DŞ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp˛
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_17/BiasAdd
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_17/Tanhp
IdentityIdentityconv2d_17/Tanh:y:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b:::::::::::::::::::::::::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs


/__inference_functional_19_layer_call_fn_6187083
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_functional_19_layer_call_and_return_conditional_losses_61870242
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
!
_user_specified_name	input_1
°
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_6188206

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_6188546
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Í
c
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_6186637
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
	
Ž
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6186542

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6185890

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulŐ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ę
`
D__inference_reshape_layer_call_and_return_conditional_losses_6188056

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3ş
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_6188246
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Š?

 __inference__traced_save_6188788
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1c269135898341dca59b0fec8cb31657/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename­
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ż
valueľB˛B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesö
ó: :	b:::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	b:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
	
­
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6186415

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ą
g
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_6188656

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_6188196
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Ţ
ĺ

J__inference_functional_19_layer_call_and_return_conditional_losses_6187901

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource
identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	b*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/BiasAddd
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape/Reshape/shape/3ę
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape 
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
reshape/ReshapeŹ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpË
conv2d/Conv2DConv2Dreshape/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d/Conv2D˘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2d/BiasAdd/ReadVariableOpĽ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d/BiasAddi
pixel_norm/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm/pow/y
pixel_norm/powPowconv2d/BiasAdd:output:0pixel_norm/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/pow
!pixel_norm/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2#
!pixel_norm/Mean/reduction_indicesľ
pixel_norm/MeanMeanpixel_norm/pow:z:0*pixel_norm/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm/Meani
pixel_norm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm/add/y
pixel_norm/addAddV2pixel_norm/Mean:output:0pixel_norm/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/addi
pixel_norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm/Constm
pixel_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm/Const_1ş
 pixel_norm/clip_by_value/MinimumMinimumpixel_norm/add:z:0pixel_norm/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 pixel_norm/clip_by_value/Minimumş
pixel_norm/clip_by_valueMaximum$pixel_norm/clip_by_value/Minimum:z:0pixel_norm/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/clip_by_value
pixel_norm/SqrtSqrtpixel_norm/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/Sqrt
pixel_norm/truedivRealDivconv2d/BiasAdd:output:0pixel_norm/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm/truediv
leaky_re_lu/LeakyRelu	LeakyRelupixel_norm/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu/LeakyRelu˛
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÜ
conv2d_1/Conv2DConv2D#leaky_re_lu/LeakyRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp­
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_1/BiasAddm
pixel_norm_1/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_1/pow/y
pixel_norm_1/powPowconv2d_1/BiasAdd:output:0pixel_norm_1/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/pow
#pixel_norm_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_1/Mean/reduction_indices˝
pixel_norm_1/MeanMeanpixel_norm_1/pow:z:0,pixel_norm_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_1/Meanm
pixel_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_1/add/y 
pixel_norm_1/addAddV2pixel_norm_1/Mean:output:0pixel_norm_1/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/addm
pixel_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_1/Constq
pixel_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_1/Const_1Â
"pixel_norm_1/clip_by_value/MinimumMinimumpixel_norm_1/add:z:0pixel_norm_1/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_1/clip_by_value/MinimumÂ
pixel_norm_1/clip_by_valueMaximum&pixel_norm_1/clip_by_value/Minimum:z:0pixel_norm_1/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/clip_by_value
pixel_norm_1/SqrtSqrtpixel_norm_1/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/Sqrt¤
pixel_norm_1/truedivRealDivconv2d_1/BiasAdd:output:0pixel_norm_1/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_1/truediv
leaky_re_lu_1/LeakyRelu	LeakyRelupixel_norm_1/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_1/LeakyRelu
up_sampling2d/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2˘
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_1/LeakyRelu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor˛
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpô
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_3/Conv2D¨
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_3/BiasAddm
pixel_norm_2/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_2/pow/y
pixel_norm_2/powPowconv2d_3/BiasAdd:output:0pixel_norm_2/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/pow
#pixel_norm_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_2/Mean/reduction_indices˝
pixel_norm_2/MeanMeanpixel_norm_2/pow:z:0,pixel_norm_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_2/Meanm
pixel_norm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_2/add/y 
pixel_norm_2/addAddV2pixel_norm_2/Mean:output:0pixel_norm_2/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/addm
pixel_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_2/Constq
pixel_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_2/Const_1Â
"pixel_norm_2/clip_by_value/MinimumMinimumpixel_norm_2/add:z:0pixel_norm_2/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_2/clip_by_value/MinimumÂ
pixel_norm_2/clip_by_valueMaximum&pixel_norm_2/clip_by_value/Minimum:z:0pixel_norm_2/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/clip_by_value
pixel_norm_2/SqrtSqrtpixel_norm_2/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/Sqrt¤
pixel_norm_2/truedivRealDivconv2d_3/BiasAdd:output:0pixel_norm_2/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_2/truediv
leaky_re_lu_2/LeakyRelu	LeakyRelupixel_norm_2/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_2/LeakyRelu˛
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpŢ
conv2d_4/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_4/BiasAddm
pixel_norm_3/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_3/pow/y
pixel_norm_3/powPowconv2d_4/BiasAdd:output:0pixel_norm_3/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/pow
#pixel_norm_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_3/Mean/reduction_indices˝
pixel_norm_3/MeanMeanpixel_norm_3/pow:z:0,pixel_norm_3/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_3/Meanm
pixel_norm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_3/add/y 
pixel_norm_3/addAddV2pixel_norm_3/Mean:output:0pixel_norm_3/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/addm
pixel_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_3/Constq
pixel_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_3/Const_1Â
"pixel_norm_3/clip_by_value/MinimumMinimumpixel_norm_3/add:z:0pixel_norm_3/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_3/clip_by_value/MinimumÂ
pixel_norm_3/clip_by_valueMaximum&pixel_norm_3/clip_by_value/Minimum:z:0pixel_norm_3/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/clip_by_value
pixel_norm_3/SqrtSqrtpixel_norm_3/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/Sqrt¤
pixel_norm_3/truedivRealDivconv2d_4/BiasAdd:output:0pixel_norm_3/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_3/truediv
leaky_re_lu_3/LeakyRelu	LeakyRelupixel_norm_3/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_3/LeakyRelu
up_sampling2d_1/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2Ž
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_3/LeakyRelu:activations:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor˛
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOpö
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_6/BiasAddm
pixel_norm_4/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_4/pow/y
pixel_norm_4/powPowconv2d_6/BiasAdd:output:0pixel_norm_4/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/pow
#pixel_norm_4/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_4/Mean/reduction_indices˝
pixel_norm_4/MeanMeanpixel_norm_4/pow:z:0,pixel_norm_4/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_4/Meanm
pixel_norm_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_4/add/y 
pixel_norm_4/addAddV2pixel_norm_4/Mean:output:0pixel_norm_4/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/addm
pixel_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_4/Constq
pixel_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_4/Const_1Â
"pixel_norm_4/clip_by_value/MinimumMinimumpixel_norm_4/add:z:0pixel_norm_4/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_4/clip_by_value/MinimumÂ
pixel_norm_4/clip_by_valueMaximum&pixel_norm_4/clip_by_value/Minimum:z:0pixel_norm_4/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/clip_by_value
pixel_norm_4/SqrtSqrtpixel_norm_4/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/Sqrt¤
pixel_norm_4/truedivRealDivconv2d_6/BiasAdd:output:0pixel_norm_4/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_4/truediv
leaky_re_lu_4/LeakyRelu	LeakyRelupixel_norm_4/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_4/LeakyRelu˛
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOpŢ
conv2d_7/Conv2DConv2D%leaky_re_lu_4/LeakyRelu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_7/Conv2D¨
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp­
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_7/BiasAddm
pixel_norm_5/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_5/pow/y
pixel_norm_5/powPowconv2d_7/BiasAdd:output:0pixel_norm_5/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/pow
#pixel_norm_5/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_5/Mean/reduction_indices˝
pixel_norm_5/MeanMeanpixel_norm_5/pow:z:0,pixel_norm_5/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_5/Meanm
pixel_norm_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_5/add/y 
pixel_norm_5/addAddV2pixel_norm_5/Mean:output:0pixel_norm_5/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/addm
pixel_norm_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_5/Constq
pixel_norm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_5/Const_1Â
"pixel_norm_5/clip_by_value/MinimumMinimumpixel_norm_5/add:z:0pixel_norm_5/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"pixel_norm_5/clip_by_value/MinimumÂ
pixel_norm_5/clip_by_valueMaximum&pixel_norm_5/clip_by_value/Minimum:z:0pixel_norm_5/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/clip_by_value
pixel_norm_5/SqrtSqrtpixel_norm_5/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/Sqrt¤
pixel_norm_5/truedivRealDivconv2d_7/BiasAdd:output:0pixel_norm_5/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_5/truediv
leaky_re_lu_5/LeakyRelu	LeakyRelupixel_norm_5/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_5/LeakyRelu
up_sampling2d_2/ShapeShape%leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2Ž
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_5/LeakyRelu:activations:0up_sampling2d_2/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor˛
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOpö
conv2d_9/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
paddingSAME*
strides
2
conv2d_9/Conv2D¨
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp­
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
conv2d_9/BiasAddm
pixel_norm_6/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_6/pow/y
pixel_norm_6/powPowconv2d_9/BiasAdd:output:0pixel_norm_6/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/pow
#pixel_norm_6/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_6/Mean/reduction_indices˝
pixel_norm_6/MeanMeanpixel_norm_6/pow:z:0,pixel_norm_6/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
	keep_dims(2
pixel_norm_6/Meanm
pixel_norm_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_6/add/y 
pixel_norm_6/addAddV2pixel_norm_6/Mean:output:0pixel_norm_6/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/addm
pixel_norm_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_6/Constq
pixel_norm_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_6/Const_1Â
"pixel_norm_6/clip_by_value/MinimumMinimumpixel_norm_6/add:z:0pixel_norm_6/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2$
"pixel_norm_6/clip_by_value/MinimumÂ
pixel_norm_6/clip_by_valueMaximum&pixel_norm_6/clip_by_value/Minimum:z:0pixel_norm_6/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/clip_by_value
pixel_norm_6/SqrtSqrtpixel_norm_6/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/Sqrt¤
pixel_norm_6/truedivRealDivconv2d_9/BiasAdd:output:0pixel_norm_6/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_6/truediv
leaky_re_lu_6/LeakyRelu	LeakyRelupixel_norm_6/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
leaky_re_lu_6/LeakyReluľ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOpá
conv2d_10/Conv2DConv2D%leaky_re_lu_6/LeakyRelu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
paddingSAME*
strides
2
conv2d_10/Conv2DŤ
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOpą
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
conv2d_10/BiasAddm
pixel_norm_7/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_7/pow/y
pixel_norm_7/powPowconv2d_10/BiasAdd:output:0pixel_norm_7/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/pow
#pixel_norm_7/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_7/Mean/reduction_indices˝
pixel_norm_7/MeanMeanpixel_norm_7/pow:z:0,pixel_norm_7/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
	keep_dims(2
pixel_norm_7/Meanm
pixel_norm_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_7/add/y 
pixel_norm_7/addAddV2pixel_norm_7/Mean:output:0pixel_norm_7/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/addm
pixel_norm_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_7/Constq
pixel_norm_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_7/Const_1Â
"pixel_norm_7/clip_by_value/MinimumMinimumpixel_norm_7/add:z:0pixel_norm_7/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2$
"pixel_norm_7/clip_by_value/MinimumÂ
pixel_norm_7/clip_by_valueMaximum&pixel_norm_7/clip_by_value/Minimum:z:0pixel_norm_7/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/clip_by_value
pixel_norm_7/SqrtSqrtpixel_norm_7/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/SqrtĽ
pixel_norm_7/truedivRealDivconv2d_10/BiasAdd:output:0pixel_norm_7/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
pixel_norm_7/truediv
leaky_re_lu_7/LeakyRelu	LeakyRelupixel_norm_7/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  2
leaky_re_lu_7/LeakyRelu
up_sampling2d_3/ShapeShape%leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2Ž
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_7/LeakyRelu:activations:0up_sampling2d_3/mul:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighborľ
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOpů
conv2d_12/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
paddingSAME*
strides
2
conv2d_12/Conv2DŤ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOpą
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
conv2d_12/BiasAddm
pixel_norm_8/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_8/pow/y
pixel_norm_8/powPowconv2d_12/BiasAdd:output:0pixel_norm_8/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/pow
#pixel_norm_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_8/Mean/reduction_indices˝
pixel_norm_8/MeanMeanpixel_norm_8/pow:z:0,pixel_norm_8/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
	keep_dims(2
pixel_norm_8/Meanm
pixel_norm_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_8/add/y 
pixel_norm_8/addAddV2pixel_norm_8/Mean:output:0pixel_norm_8/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/addm
pixel_norm_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_8/Constq
pixel_norm_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_8/Const_1Â
"pixel_norm_8/clip_by_value/MinimumMinimumpixel_norm_8/add:z:0pixel_norm_8/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2$
"pixel_norm_8/clip_by_value/MinimumÂ
pixel_norm_8/clip_by_valueMaximum&pixel_norm_8/clip_by_value/Minimum:z:0pixel_norm_8/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/clip_by_value
pixel_norm_8/SqrtSqrtpixel_norm_8/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/SqrtĽ
pixel_norm_8/truedivRealDivconv2d_12/BiasAdd:output:0pixel_norm_8/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_8/truediv
leaky_re_lu_8/LeakyRelu	LeakyRelupixel_norm_8/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
leaky_re_lu_8/LeakyReluľ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOpá
conv2d_13/Conv2DConv2D%leaky_re_lu_8/LeakyRelu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
paddingSAME*
strides
2
conv2d_13/Conv2DŤ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOpą
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
conv2d_13/BiasAddm
pixel_norm_9/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_9/pow/y
pixel_norm_9/powPowconv2d_13/BiasAdd:output:0pixel_norm_9/pow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/pow
#pixel_norm_9/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2%
#pixel_norm_9/Mean/reduction_indices˝
pixel_norm_9/MeanMeanpixel_norm_9/pow:z:0,pixel_norm_9/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@*
	keep_dims(2
pixel_norm_9/Meanm
pixel_norm_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_9/add/y 
pixel_norm_9/addAddV2pixel_norm_9/Mean:output:0pixel_norm_9/add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/addm
pixel_norm_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_9/Constq
pixel_norm_9/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_9/Const_1Â
"pixel_norm_9/clip_by_value/MinimumMinimumpixel_norm_9/add:z:0pixel_norm_9/Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2$
"pixel_norm_9/clip_by_value/MinimumÂ
pixel_norm_9/clip_by_valueMaximum&pixel_norm_9/clip_by_value/Minimum:z:0pixel_norm_9/Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/clip_by_value
pixel_norm_9/SqrtSqrtpixel_norm_9/clip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/SqrtĽ
pixel_norm_9/truedivRealDivconv2d_13/BiasAdd:output:0pixel_norm_9/Sqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
pixel_norm_9/truediv
leaky_re_lu_9/LeakyRelu	LeakyRelupixel_norm_9/truediv:z:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@2
leaky_re_lu_9/LeakyRelu
up_sampling2d_4/ShapeShape%leaky_re_lu_9/LeakyRelu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2Ž
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor%leaky_re_lu_9/LeakyRelu:activations:0up_sampling2d_4/mul:z:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighborľ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOpű
conv2d_15/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_15/Conv2DŤ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpł
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
conv2d_15/BiasAddo
pixel_norm_10/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_10/pow/y¤
pixel_norm_10/powPowconv2d_15/BiasAdd:output:0pixel_norm_10/pow/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/pow
$pixel_norm_10/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2&
$pixel_norm_10/Mean/reduction_indicesĂ
pixel_norm_10/MeanMeanpixel_norm_10/pow:z:0-pixel_norm_10/Mean/reduction_indices:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_10/Meano
pixel_norm_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_10/add/yŚ
pixel_norm_10/addAddV2pixel_norm_10/Mean:output:0pixel_norm_10/add/y:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/addo
pixel_norm_10/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_10/Consts
pixel_norm_10/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_10/Const_1Č
#pixel_norm_10/clip_by_value/MinimumMinimumpixel_norm_10/add:z:0pixel_norm_10/Const_1:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#pixel_norm_10/clip_by_value/MinimumČ
pixel_norm_10/clip_by_valueMaximum'pixel_norm_10/clip_by_value/Minimum:z:0pixel_norm_10/Const:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/clip_by_value
pixel_norm_10/SqrtSqrtpixel_norm_10/clip_by_value:z:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/SqrtŞ
pixel_norm_10/truedivRealDivconv2d_15/BiasAdd:output:0pixel_norm_10/Sqrt:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_10/truediv
leaky_re_lu_10/LeakyRelu	LeakyRelupixel_norm_10/truediv:z:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_10/LeakyReluľ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_16/Conv2D/ReadVariableOpä
conv2d_16/Conv2DConv2D&leaky_re_lu_10/LeakyRelu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_16/Conv2DŤ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOpł
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
conv2d_16/BiasAddo
pixel_norm_11/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pixel_norm_11/pow/y¤
pixel_norm_11/powPowconv2d_16/BiasAdd:output:0pixel_norm_11/pow/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/pow
$pixel_norm_11/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2&
$pixel_norm_11/Mean/reduction_indicesĂ
pixel_norm_11/MeanMeanpixel_norm_11/pow:z:0-pixel_norm_11/Mean/reduction_indices:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
pixel_norm_11/Meano
pixel_norm_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
pixel_norm_11/add/yŚ
pixel_norm_11/addAddV2pixel_norm_11/Mean:output:0pixel_norm_11/add/y:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/addo
pixel_norm_11/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
pixel_norm_11/Consts
pixel_norm_11/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2
pixel_norm_11/Const_1Č
#pixel_norm_11/clip_by_value/MinimumMinimumpixel_norm_11/add:z:0pixel_norm_11/Const_1:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#pixel_norm_11/clip_by_value/MinimumČ
pixel_norm_11/clip_by_valueMaximum'pixel_norm_11/clip_by_value/Minimum:z:0pixel_norm_11/Const:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/clip_by_value
pixel_norm_11/SqrtSqrtpixel_norm_11/clip_by_value:z:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/SqrtŞ
pixel_norm_11/truedivRealDivconv2d_16/BiasAdd:output:0pixel_norm_11/Sqrt:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
pixel_norm_11/truediv
leaky_re_lu_11/LeakyRelu	LeakyRelupixel_norm_11/truediv:z:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
leaky_re_lu_11/LeakyRelu´
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_17/Conv2D/ReadVariableOpă
conv2d_17/Conv2DConv2D&leaky_re_lu_11/LeakyRelu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
conv2d_17/Conv2DŞ
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp˛
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_17/BiasAdd
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_17/Tanhp
IdentityIdentityconv2d_17/Tanh:y:0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b:::::::::::::::::::::::::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
Š
E
)__inference_reshape_layer_call_fn_6188061

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_61860162
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6186351

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ž
F__inference_conv2d_16_layer_call_and_return_conditional_losses_6186732

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ź
Ť
C__inference_conv2d_layer_call_and_return_conditional_losses_6188071

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6188271

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
d
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_6186701
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
đ	
Ž
F__inference_conv2d_17_layer_call_and_return_conditional_losses_6188672

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpľ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Tanhv
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_6186574
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata


/__inference_functional_19_layer_call_fn_6187962

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_functional_19_layer_call_and_return_conditional_losses_61870242
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
˛
M
1__inference_up_sampling2d_2_layer_call_fn_6185934

inputs
identityđ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_61859282
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĺ
d
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6186079

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_6186193
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

K
/__inference_leaky_re_lu_8_layer_call_fn_6188511

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_61865872
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_6186206

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

c
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_6186129
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yb
powPowdatapow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/yl
addAddV2Mean:output:0add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_valuea
SqrtSqrtclip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sqrth
truedivRealDivdataSqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivh
IdentityIdentitytruediv:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:V R
0
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namedata
	
Ž
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6186605

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6186288

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î
d
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_6186764
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata

h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6185909

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulŐ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6188221

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ž
F__inference_conv2d_15_layer_call_and_return_conditional_losses_6186669

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_6188256

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


/__inference_functional_19_layer_call_fn_6187248
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_functional_19_layer_call_and_return_conditional_losses_61871892
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
!
_user_specified_name	input_1

K
/__inference_leaky_re_lu_3_layer_call_fn_6188261

inputs
identityć
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_61862692
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ĺ
K
/__inference_leaky_re_lu_1_layer_call_fn_6188161

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_61861422
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ž
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6186478

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

H
.__inference_pixel_norm_6_layer_call_fn_6188401
data
identityă
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_61864472
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
˝
H
.__inference_pixel_norm_1_layer_call_fn_6188151
data
identityŃ
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_61861292
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:V R
0
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Í

*__inference_conv2d_3_layer_call_fn_6188180

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_61861612
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

I
/__inference_pixel_norm_10_layer_call_fn_6188601
data
identityä
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_61867012
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
ą
g
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_6188606

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í
c
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_6186320
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Í
c
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_6188446
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yt
powPowdatapow/y:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/y~
addAddV2Mean:output:0add/y:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1 
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum 
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
clip_by_values
SqrtSqrtclip_by_value:z:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Sqrtz
truedivRealDivdataSqrt:y:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
truedivz
IdentityIdentitytruediv:z:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
ę
`
D__inference_reshape_layer_call_and_return_conditional_losses_6186016

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3ş
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_6186396

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
t
ü
#__inference__traced_restore_6188882
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias$
 assignvariableop_2_conv2d_kernel"
assignvariableop_3_conv2d_bias&
"assignvariableop_4_conv2d_1_kernel$
 assignvariableop_5_conv2d_1_bias&
"assignvariableop_6_conv2d_3_kernel$
 assignvariableop_7_conv2d_3_bias&
"assignvariableop_8_conv2d_4_kernel$
 assignvariableop_9_conv2d_4_bias'
#assignvariableop_10_conv2d_6_kernel%
!assignvariableop_11_conv2d_6_bias'
#assignvariableop_12_conv2d_7_kernel%
!assignvariableop_13_conv2d_7_bias'
#assignvariableop_14_conv2d_9_kernel%
!assignvariableop_15_conv2d_9_bias(
$assignvariableop_16_conv2d_10_kernel&
"assignvariableop_17_conv2d_10_bias(
$assignvariableop_18_conv2d_12_kernel&
"assignvariableop_19_conv2d_12_bias(
$assignvariableop_20_conv2d_13_kernel&
"assignvariableop_21_conv2d_13_bias(
$assignvariableop_22_conv2d_15_kernel&
"assignvariableop_23_conv2d_15_bias(
$assignvariableop_24_conv2d_16_kernel&
"assignvariableop_25_conv2d_16_bias(
$assignvariableop_26_conv2d_17_kernel&
"assignvariableop_27_conv2d_17_bias
identity_29˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9ł
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ż
valueľB˛B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesČ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices˝
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1˘
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ľ
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ł
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ľ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ľ
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ľ
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ť
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Š
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ť
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Š
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ť
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_9_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Š
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_9_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ź
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_10_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ş
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_10_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ź
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_12_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ş
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_12_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ź
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_13_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ş
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_13_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ź
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_15_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ş
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_15_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ź
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_16_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ş
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_16_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ź
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_17_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ş
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_17_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpĆ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28š
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*
_input_shapest
r: ::::::::::::::::::::::::::::2$
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
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°
f
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_6186523

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_6186333

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
­
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6186224

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
­
ż	
J__inference_functional_19_layer_call_and_return_conditional_losses_6186813
input_1
dense_6185997
dense_6185999
conv2d_6186045
conv2d_6186047
conv2d_1_6186108
conv2d_1_6186110
conv2d_3_6186172
conv2d_3_6186174
conv2d_4_6186235
conv2d_4_6186237
conv2d_6_6186299
conv2d_6_6186301
conv2d_7_6186362
conv2d_7_6186364
conv2d_9_6186426
conv2d_9_6186428
conv2d_10_6186489
conv2d_10_6186491
conv2d_12_6186553
conv2d_12_6186555
conv2d_13_6186616
conv2d_13_6186618
conv2d_15_6186680
conv2d_15_6186682
conv2d_16_6186743
conv2d_16_6186745
conv2d_17_6186807
conv2d_17_6186809
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘!conv2d_10/StatefulPartitionedCall˘!conv2d_12/StatefulPartitionedCall˘!conv2d_13/StatefulPartitionedCall˘!conv2d_15/StatefulPartitionedCall˘!conv2d_16/StatefulPartitionedCall˘!conv2d_17/StatefulPartitionedCall˘ conv2d_3/StatefulPartitionedCall˘ conv2d_4/StatefulPartitionedCall˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_9/StatefulPartitionedCall˘dense/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6185997dense_6185999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_61859862
dense/StatefulPartitionedCallţ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_61860162
reshape/PartitionedCallł
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_6186045conv2d_6186047*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_61860342 
conv2d/StatefulPartitionedCall
pixel_norm/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_pixel_norm_layer_call_and_return_conditional_losses_61860662
pixel_norm/PartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall#pixel_norm/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_61860792
leaky_re_lu/PartitionedCallÁ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv2d_1_6186108conv2d_1_6186110*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_61860972"
 conv2d_1/StatefulPartitionedCall
pixel_norm_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_61861292
pixel_norm_1/PartitionedCall
leaky_re_lu_1/PartitionedCallPartitionedCall%pixel_norm_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_61861422
leaky_re_lu_1/PartitionedCall˘
up_sampling2d/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_61858902
up_sampling2d/PartitionedCallŐ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_6186172conv2d_3_6186174*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_61861612"
 conv2d_3/StatefulPartitionedCall˘
pixel_norm_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_61861932
pixel_norm_2/PartitionedCallĄ
leaky_re_lu_2/PartitionedCallPartitionedCall%pixel_norm_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_61862062
leaky_re_lu_2/PartitionedCallŐ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv2d_4_6186235conv2d_4_6186237*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_61862242"
 conv2d_4/StatefulPartitionedCall˘
pixel_norm_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_61862562
pixel_norm_3/PartitionedCallĄ
leaky_re_lu_3/PartitionedCallPartitionedCall%pixel_norm_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_61862692
leaky_re_lu_3/PartitionedCall¨
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_61859092!
up_sampling2d_1/PartitionedCall×
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_6186299conv2d_6_6186301*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_61862882"
 conv2d_6/StatefulPartitionedCall˘
pixel_norm_4/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_61863202
pixel_norm_4/PartitionedCallĄ
leaky_re_lu_4/PartitionedCallPartitionedCall%pixel_norm_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_61863332
leaky_re_lu_4/PartitionedCallŐ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv2d_7_6186362conv2d_7_6186364*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_61863512"
 conv2d_7/StatefulPartitionedCall˘
pixel_norm_5/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_61863832
pixel_norm_5/PartitionedCallĄ
leaky_re_lu_5/PartitionedCallPartitionedCall%pixel_norm_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_61863962
leaky_re_lu_5/PartitionedCall¨
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_61859282!
up_sampling2d_2/PartitionedCall×
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_9_6186426conv2d_9_6186428*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_61864152"
 conv2d_9/StatefulPartitionedCall˘
pixel_norm_6/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_61864472
pixel_norm_6/PartitionedCallĄ
leaky_re_lu_6/PartitionedCallPartitionedCall%pixel_norm_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_61864602
leaky_re_lu_6/PartitionedCallÚ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0conv2d_10_6186489conv2d_10_6186491*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_61864782#
!conv2d_10/StatefulPartitionedCallŁ
pixel_norm_7/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_61865102
pixel_norm_7/PartitionedCallĄ
leaky_re_lu_7/PartitionedCallPartitionedCall%pixel_norm_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_61865232
leaky_re_lu_7/PartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_61859472!
up_sampling2d_3/PartitionedCallÜ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_12_6186553conv2d_12_6186555*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_61865422#
!conv2d_12/StatefulPartitionedCallŁ
pixel_norm_8/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_61865742
pixel_norm_8/PartitionedCallĄ
leaky_re_lu_8/PartitionedCallPartitionedCall%pixel_norm_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_61865872
leaky_re_lu_8/PartitionedCallÚ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0conv2d_13_6186616conv2d_13_6186618*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_61866052#
!conv2d_13/StatefulPartitionedCallŁ
pixel_norm_9/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_61866372
pixel_norm_9/PartitionedCallĄ
leaky_re_lu_9/PartitionedCallPartitionedCall%pixel_norm_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_61866502
leaky_re_lu_9/PartitionedCall¨
up_sampling2d_4/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_61859662!
up_sampling2d_4/PartitionedCallÜ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_15_6186680conv2d_15_6186682*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_61866692#
!conv2d_15/StatefulPartitionedCallŚ
pixel_norm_10/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_61867012
pixel_norm_10/PartitionedCallĽ
leaky_re_lu_10/PartitionedCallPartitionedCall&pixel_norm_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_61867142 
leaky_re_lu_10/PartitionedCallŰ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0conv2d_16_6186743conv2d_16_6186745*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_61867322#
!conv2d_16/StatefulPartitionedCallŚ
pixel_norm_11/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_61867642
pixel_norm_11/PartitionedCallĽ
leaky_re_lu_11/PartitionedCallPartitionedCall&pixel_norm_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_61867772 
leaky_re_lu_11/PartitionedCallÚ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0conv2d_17_6186807conv2d_17_6186809*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_61867962#
!conv2d_17/StatefulPartitionedCall
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
!
_user_specified_name	input_1

L
0__inference_leaky_re_lu_11_layer_call_fn_6188661

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_61867772
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
­
ž	
J__inference_functional_19_layer_call_and_return_conditional_losses_6187189

inputs
dense_6187088
dense_6187090
conv2d_6187094
conv2d_6187096
conv2d_1_6187101
conv2d_1_6187103
conv2d_3_6187109
conv2d_3_6187111
conv2d_4_6187116
conv2d_4_6187118
conv2d_6_6187124
conv2d_6_6187126
conv2d_7_6187131
conv2d_7_6187133
conv2d_9_6187139
conv2d_9_6187141
conv2d_10_6187146
conv2d_10_6187148
conv2d_12_6187154
conv2d_12_6187156
conv2d_13_6187161
conv2d_13_6187163
conv2d_15_6187169
conv2d_15_6187171
conv2d_16_6187176
conv2d_16_6187178
conv2d_17_6187183
conv2d_17_6187185
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘!conv2d_10/StatefulPartitionedCall˘!conv2d_12/StatefulPartitionedCall˘!conv2d_13/StatefulPartitionedCall˘!conv2d_15/StatefulPartitionedCall˘!conv2d_16/StatefulPartitionedCall˘!conv2d_17/StatefulPartitionedCall˘ conv2d_3/StatefulPartitionedCall˘ conv2d_4/StatefulPartitionedCall˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_9/StatefulPartitionedCall˘dense/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6187088dense_6187090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_61859862
dense/StatefulPartitionedCallţ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_61860162
reshape/PartitionedCallł
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_6187094conv2d_6187096*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_61860342 
conv2d/StatefulPartitionedCall
pixel_norm/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_pixel_norm_layer_call_and_return_conditional_losses_61860662
pixel_norm/PartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall#pixel_norm/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_61860792
leaky_re_lu/PartitionedCallÁ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv2d_1_6187101conv2d_1_6187103*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_61860972"
 conv2d_1/StatefulPartitionedCall
pixel_norm_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_61861292
pixel_norm_1/PartitionedCall
leaky_re_lu_1/PartitionedCallPartitionedCall%pixel_norm_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_61861422
leaky_re_lu_1/PartitionedCall˘
up_sampling2d/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_61858902
up_sampling2d/PartitionedCallŐ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_6187109conv2d_3_6187111*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_61861612"
 conv2d_3/StatefulPartitionedCall˘
pixel_norm_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_61861932
pixel_norm_2/PartitionedCallĄ
leaky_re_lu_2/PartitionedCallPartitionedCall%pixel_norm_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_61862062
leaky_re_lu_2/PartitionedCallŐ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv2d_4_6187116conv2d_4_6187118*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_61862242"
 conv2d_4/StatefulPartitionedCall˘
pixel_norm_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_61862562
pixel_norm_3/PartitionedCallĄ
leaky_re_lu_3/PartitionedCallPartitionedCall%pixel_norm_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_61862692
leaky_re_lu_3/PartitionedCall¨
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_61859092!
up_sampling2d_1/PartitionedCall×
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_6187124conv2d_6_6187126*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_61862882"
 conv2d_6/StatefulPartitionedCall˘
pixel_norm_4/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_61863202
pixel_norm_4/PartitionedCallĄ
leaky_re_lu_4/PartitionedCallPartitionedCall%pixel_norm_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_61863332
leaky_re_lu_4/PartitionedCallŐ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv2d_7_6187131conv2d_7_6187133*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_61863512"
 conv2d_7/StatefulPartitionedCall˘
pixel_norm_5/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_61863832
pixel_norm_5/PartitionedCallĄ
leaky_re_lu_5/PartitionedCallPartitionedCall%pixel_norm_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_61863962
leaky_re_lu_5/PartitionedCall¨
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_61859282!
up_sampling2d_2/PartitionedCall×
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_9_6187139conv2d_9_6187141*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_61864152"
 conv2d_9/StatefulPartitionedCall˘
pixel_norm_6/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_61864472
pixel_norm_6/PartitionedCallĄ
leaky_re_lu_6/PartitionedCallPartitionedCall%pixel_norm_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_61864602
leaky_re_lu_6/PartitionedCallÚ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0conv2d_10_6187146conv2d_10_6187148*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_61864782#
!conv2d_10/StatefulPartitionedCallŁ
pixel_norm_7/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_61865102
pixel_norm_7/PartitionedCallĄ
leaky_re_lu_7/PartitionedCallPartitionedCall%pixel_norm_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_61865232
leaky_re_lu_7/PartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_61859472!
up_sampling2d_3/PartitionedCallÜ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_12_6187154conv2d_12_6187156*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_61865422#
!conv2d_12/StatefulPartitionedCallŁ
pixel_norm_8/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_61865742
pixel_norm_8/PartitionedCallĄ
leaky_re_lu_8/PartitionedCallPartitionedCall%pixel_norm_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_61865872
leaky_re_lu_8/PartitionedCallÚ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0conv2d_13_6187161conv2d_13_6187163*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_61866052#
!conv2d_13/StatefulPartitionedCallŁ
pixel_norm_9/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_61866372
pixel_norm_9/PartitionedCallĄ
leaky_re_lu_9/PartitionedCallPartitionedCall%pixel_norm_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_61866502
leaky_re_lu_9/PartitionedCall¨
up_sampling2d_4/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_61859662!
up_sampling2d_4/PartitionedCallÜ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_15_6187169conv2d_15_6187171*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_61866692#
!conv2d_15/StatefulPartitionedCallŚ
pixel_norm_10/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_61867012
pixel_norm_10/PartitionedCallĽ
leaky_re_lu_10/PartitionedCallPartitionedCall&pixel_norm_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_61867142 
leaky_re_lu_10/PartitionedCallŰ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0conv2d_16_6187176conv2d_16_6187178*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_61867322#
!conv2d_16/StatefulPartitionedCallŚ
pixel_norm_11/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_61867642
pixel_norm_11/PartitionedCallĽ
leaky_re_lu_11/PartitionedCallPartitionedCall&pixel_norm_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_61867772 
leaky_re_lu_11/PartitionedCallÚ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0conv2d_17_6187183conv2d_17_6187185*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_61867962#
!conv2d_17/StatefulPartitionedCall
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
Đ

+__inference_conv2d_15_layer_call_fn_6188580

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_61866692
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_6186650

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ź
Ť
C__inference_conv2d_layer_call_and_return_conditional_losses_6186034

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
Ş
B__inference_dense_layer_call_and_return_conditional_losses_6188033

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	b*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙b:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
Á
I
-__inference_leaky_re_lu_layer_call_fn_6188111

inputs
identityŇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_61860792
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Ž
F__inference_conv2d_15_layer_call_and_return_conditional_losses_6188571

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpś
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

I
/__inference_pixel_norm_11_layer_call_fn_6188651
data
identityä
PartitionedCallPartitionedCalldata*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_61867642
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:h d
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Ž
­
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6186097

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í

*__inference_conv2d_9_layer_call_fn_6188380

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_61864152
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Í

*__inference_conv2d_7_layer_call_fn_6188330

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_61863512
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°
f
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_6188356

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ą
g
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_6186714

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

a
G__inference_pixel_norm_layer_call_and_return_conditional_losses_6188096
data
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yb
powPowdatapow/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2
MeanS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+22
add/yl
addAddV2Mean:output:0add/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  2	
Const_1
clip_by_value/MinimumMinimumadd:z:0Const_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_value/Minimum
clip_by_valueMaximumclip_by_value/Minimum:z:0Const:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
clip_by_valuea
SqrtSqrtclip_by_value:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Sqrth
truedivRealDivdataSqrt:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
truedivh
IdentityIdentitytruediv:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:V R
0
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namedata
Ý
|
'__inference_dense_layer_call_fn_6188042

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_61859862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙b::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
 
_user_specified_nameinputs
­
ż	
J__inference_functional_19_layer_call_and_return_conditional_losses_6186917
input_1
dense_6186816
dense_6186818
conv2d_6186822
conv2d_6186824
conv2d_1_6186829
conv2d_1_6186831
conv2d_3_6186837
conv2d_3_6186839
conv2d_4_6186844
conv2d_4_6186846
conv2d_6_6186852
conv2d_6_6186854
conv2d_7_6186859
conv2d_7_6186861
conv2d_9_6186867
conv2d_9_6186869
conv2d_10_6186874
conv2d_10_6186876
conv2d_12_6186882
conv2d_12_6186884
conv2d_13_6186889
conv2d_13_6186891
conv2d_15_6186897
conv2d_15_6186899
conv2d_16_6186904
conv2d_16_6186906
conv2d_17_6186911
conv2d_17_6186913
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘!conv2d_10/StatefulPartitionedCall˘!conv2d_12/StatefulPartitionedCall˘!conv2d_13/StatefulPartitionedCall˘!conv2d_15/StatefulPartitionedCall˘!conv2d_16/StatefulPartitionedCall˘!conv2d_17/StatefulPartitionedCall˘ conv2d_3/StatefulPartitionedCall˘ conv2d_4/StatefulPartitionedCall˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_9/StatefulPartitionedCall˘dense/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6186816dense_6186818*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_61859862
dense/StatefulPartitionedCallţ
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_61860162
reshape/PartitionedCallł
conv2d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_6186822conv2d_6186824*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_61860342 
conv2d/StatefulPartitionedCall
pixel_norm/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_pixel_norm_layer_call_and_return_conditional_losses_61860662
pixel_norm/PartitionedCall
leaky_re_lu/PartitionedCallPartitionedCall#pixel_norm/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_61860792
leaky_re_lu/PartitionedCallÁ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv2d_1_6186829conv2d_1_6186831*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_61860972"
 conv2d_1/StatefulPartitionedCall
pixel_norm_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_61861292
pixel_norm_1/PartitionedCall
leaky_re_lu_1/PartitionedCallPartitionedCall%pixel_norm_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_61861422
leaky_re_lu_1/PartitionedCall˘
up_sampling2d/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_61858902
up_sampling2d/PartitionedCallŐ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_6186837conv2d_3_6186839*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_61861612"
 conv2d_3/StatefulPartitionedCall˘
pixel_norm_2/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_61861932
pixel_norm_2/PartitionedCallĄ
leaky_re_lu_2/PartitionedCallPartitionedCall%pixel_norm_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_61862062
leaky_re_lu_2/PartitionedCallŐ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0conv2d_4_6186844conv2d_4_6186846*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_61862242"
 conv2d_4/StatefulPartitionedCall˘
pixel_norm_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_61862562
pixel_norm_3/PartitionedCallĄ
leaky_re_lu_3/PartitionedCallPartitionedCall%pixel_norm_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_61862692
leaky_re_lu_3/PartitionedCall¨
up_sampling2d_1/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_61859092!
up_sampling2d_1/PartitionedCall×
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_6186852conv2d_6_6186854*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_61862882"
 conv2d_6/StatefulPartitionedCall˘
pixel_norm_4/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_61863202
pixel_norm_4/PartitionedCallĄ
leaky_re_lu_4/PartitionedCallPartitionedCall%pixel_norm_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_61863332
leaky_re_lu_4/PartitionedCallŐ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0conv2d_7_6186859conv2d_7_6186861*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_61863512"
 conv2d_7/StatefulPartitionedCall˘
pixel_norm_5/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_61863832
pixel_norm_5/PartitionedCallĄ
leaky_re_lu_5/PartitionedCallPartitionedCall%pixel_norm_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_61863962
leaky_re_lu_5/PartitionedCall¨
up_sampling2d_2/PartitionedCallPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_61859282!
up_sampling2d_2/PartitionedCall×
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_9_6186867conv2d_9_6186869*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_61864152"
 conv2d_9/StatefulPartitionedCall˘
pixel_norm_6/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_61864472
pixel_norm_6/PartitionedCallĄ
leaky_re_lu_6/PartitionedCallPartitionedCall%pixel_norm_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_61864602
leaky_re_lu_6/PartitionedCallÚ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0conv2d_10_6186874conv2d_10_6186876*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_61864782#
!conv2d_10/StatefulPartitionedCallŁ
pixel_norm_7/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_61865102
pixel_norm_7/PartitionedCallĄ
leaky_re_lu_7/PartitionedCallPartitionedCall%pixel_norm_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_61865232
leaky_re_lu_7/PartitionedCall¨
up_sampling2d_3/PartitionedCallPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_61859472!
up_sampling2d_3/PartitionedCallÜ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_12_6186882conv2d_12_6186884*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_61865422#
!conv2d_12/StatefulPartitionedCallŁ
pixel_norm_8/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_61865742
pixel_norm_8/PartitionedCallĄ
leaky_re_lu_8/PartitionedCallPartitionedCall%pixel_norm_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_61865872
leaky_re_lu_8/PartitionedCallÚ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0conv2d_13_6186889conv2d_13_6186891*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_61866052#
!conv2d_13/StatefulPartitionedCallŁ
pixel_norm_9/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_61866372
pixel_norm_9/PartitionedCallĄ
leaky_re_lu_9/PartitionedCallPartitionedCall%pixel_norm_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_61866502
leaky_re_lu_9/PartitionedCall¨
up_sampling2d_4/PartitionedCallPartitionedCall&leaky_re_lu_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_61859662!
up_sampling2d_4/PartitionedCallÜ
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_15_6186897conv2d_15_6186899*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_61866692#
!conv2d_15/StatefulPartitionedCallŚ
pixel_norm_10/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_61867012
pixel_norm_10/PartitionedCallĽ
leaky_re_lu_10/PartitionedCallPartitionedCall&pixel_norm_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_61867142 
leaky_re_lu_10/PartitionedCallŰ
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_10/PartitionedCall:output:0conv2d_16_6186904conv2d_16_6186906*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_16_layer_call_and_return_conditional_losses_61867322#
!conv2d_16/StatefulPartitionedCallŚ
pixel_norm_11/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_61867642
pixel_norm_11/PartitionedCallĽ
leaky_re_lu_11/PartitionedCallPartitionedCall&pixel_norm_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_61867772 
leaky_re_lu_11/PartitionedCallÚ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_11/PartitionedCall:output:0conv2d_17_6186911conv2d_17_6186913*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_17_layer_call_and_return_conditional_losses_61867962#
!conv2d_17/StatefulPartitionedCall
IdentityIdentity*conv2d_17/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*
_input_shapes
:˙˙˙˙˙˙˙˙˙b::::::::::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
!
_user_specified_name	input_1"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ś
serving_default˘
;
input_10
serving_default_input_1:0˙˙˙˙˙˙˙˙˙bG
	conv2d_17:
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:űÜ
˝§
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer-16
layer_with_weights-5
layer-17
layer-18
layer-19
layer_with_weights-6
layer-20
layer-21
layer-22
layer-23
layer_with_weights-7
layer-24
layer-25
layer-26
layer_with_weights-8
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-9
 layer-31
!layer-32
"layer-33
#layer_with_weights-10
#layer-34
$layer-35
%layer-36
&layer-37
'layer_with_weights-11
'layer-38
(layer-39
)layer-40
*layer_with_weights-12
*layer-41
+layer-42
,layer-43
-layer_with_weights-13
-layer-44
.trainable_variables
/	variables
0regularization_losses
1	keras_api
2
signatures
+ŕ&call_and_return_all_conditional_losses
á_default_save_signature
â__call__"Ą
_tf_keras_network{"class_name": "Functional", "name": "functional_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 98]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2048, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 128]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm", "trainable": true, "dtype": "float32"}, "name": "pixel_norm", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["pixel_norm", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_1", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["pixel_norm_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_2", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_2", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["pixel_norm_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_3", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_3", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["pixel_norm_3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_4", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_4", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["pixel_norm_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_5", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_5", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["pixel_norm_5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_6", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_6", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["pixel_norm_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_7", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_7", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["pixel_norm_7", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_8", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_8", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_8", "inbound_nodes": [[["pixel_norm_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_9", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_9", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_9", "inbound_nodes": [[["pixel_norm_9", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_10", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_10", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["pixel_norm_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_11", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_11", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_11", "inbound_nodes": [[["pixel_norm_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["leaky_re_lu_11", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_17", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 98]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 98]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2048, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 128]}}, "name": "reshape", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm", "trainable": true, "dtype": "float32"}, "name": "pixel_norm", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu", "inbound_nodes": [[["pixel_norm", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["leaky_re_lu", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_1", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_1", "inbound_nodes": [[["pixel_norm_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_2", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_2", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_2", "inbound_nodes": [[["pixel_norm_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_3", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_3", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_3", "inbound_nodes": [[["pixel_norm_3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_4", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_4", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["pixel_norm_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_5", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_5", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["pixel_norm_5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_6", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_6", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_6", "inbound_nodes": [[["pixel_norm_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_7", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_7", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_7", "inbound_nodes": [[["pixel_norm_7", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_8", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_8", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_8", "inbound_nodes": [[["pixel_norm_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_9", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_9", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_9", "inbound_nodes": [[["pixel_norm_9", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_10", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_10", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_10", "inbound_nodes": [[["pixel_norm_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]]}, {"class_name": "PixelNorm", "config": {"name": "pixel_norm_11", "trainable": true, "dtype": "float32"}, "name": "pixel_norm_11", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_11", "inbound_nodes": [[["pixel_norm_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["leaky_re_lu_11", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_17", 0, 0]]}}}
ë"č
_tf_keras_input_layerČ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 98]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 98]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Ě

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+ă&call_and_return_all_conditional_losses
ä__call__"Ľ
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2048, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 98}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 98]}}
÷
9trainable_variables
:	variables
;regularization_losses
<	keras_api
+ĺ&call_and_return_all_conditional_losses
ć__call__"ć
_tf_keras_layerĚ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 128]}}}
Í


=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
+ç&call_and_return_all_conditional_losses
č__call__"Ś	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
ź
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
+é&call_and_return_all_conditional_losses
ę__call__"Ť
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm", "trainable": true, "dtype": "float32"}}
Ü
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
+ë&call_and_return_all_conditional_losses
ě__call__"Ë
_tf_keras_layerą{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ń


Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
+í&call_and_return_all_conditional_losses
î__call__"Ş	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
Ŕ
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
+ď&call_and_return_all_conditional_losses
đ__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_1", "trainable": true, "dtype": "float32"}}
ŕ
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
+ń&call_and_return_all_conditional_losses
ň__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ç
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"ś
_tf_keras_layer{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ń


]kernel
^bias
_trainable_variables
`	variables
aregularization_losses
b	keras_api
+ő&call_and_return_all_conditional_losses
ö__call__"Ş	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
Ŕ
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
+÷&call_and_return_all_conditional_losses
ř__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_2", "trainable": true, "dtype": "float32"}}
ŕ
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
+ů&call_and_return_all_conditional_losses
ú__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ń


kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+ű&call_and_return_all_conditional_losses
ü__call__"Ş	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
Ŕ
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
+ý&call_and_return_all_conditional_losses
ţ__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_3", "trainable": true, "dtype": "float32"}}
ŕ
utrainable_variables
v	variables
wregularization_losses
x	keras_api
+˙&call_and_return_all_conditional_losses
__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ë
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
+&call_and_return_all_conditional_losses
__call__"ş
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ö


}kernel
~bias
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ź	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
Ä
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_4", "trainable": true, "dtype": "float32"}}
ä
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ů

kernel
	bias
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ź	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
Ä
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_5", "trainable": true, "dtype": "float32"}}
ä
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ď
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ş
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ů

kernel
	bias
trainable_variables
 	variables
Ąregularization_losses
˘	keras_api
+&call_and_return_all_conditional_losses
__call__"Ź	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Ä
Łtrainable_variables
¤	variables
Ľregularization_losses
Ś	keras_api
+&call_and_return_all_conditional_losses
__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_6", "trainable": true, "dtype": "float32"}}
ä
§trainable_variables
¨	variables
Šregularization_losses
Ş	keras_api
+&call_and_return_all_conditional_losses
__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ű

Ťkernel
	Źbias
­trainable_variables
Ž	variables
Żregularization_losses
°	keras_api
+&call_and_return_all_conditional_losses
__call__"Ž	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
Ä
ątrainable_variables
˛	variables
łregularization_losses
´	keras_api
+&call_and_return_all_conditional_losses
__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_7", "trainable": true, "dtype": "float32"}}
ä
ľtrainable_variables
ś	variables
ˇregularization_losses
¸	keras_api
+&call_and_return_all_conditional_losses
__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ď
štrainable_variables
ş	variables
ťregularization_losses
ź	keras_api
+&call_and_return_all_conditional_losses
__call__"ş
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ű

˝kernel
	žbias
żtrainable_variables
Ŕ	variables
Áregularization_losses
Â	keras_api
+&call_and_return_all_conditional_losses
 __call__"Ž	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}
Ä
Ătrainable_variables
Ä	variables
Ĺregularization_losses
Ć	keras_api
+Ą&call_and_return_all_conditional_losses
˘__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_8", "trainable": true, "dtype": "float32"}}
ä
Çtrainable_variables
Č	variables
Éregularization_losses
Ę	keras_api
+Ł&call_and_return_all_conditional_losses
¤__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ű

Ëkernel
	Ěbias
Ítrainable_variables
Î	variables
Ďregularization_losses
Đ	keras_api
+Ľ&call_and_return_all_conditional_losses
Ś__call__"Ž	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}
Ä
Ńtrainable_variables
Ň	variables
Óregularization_losses
Ô	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"Ż
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_9", "trainable": true, "dtype": "float32"}}
ä
Őtrainable_variables
Ö	variables
×regularization_losses
Ř	keras_api
+Š&call_and_return_all_conditional_losses
Ş__call__"Ď
_tf_keras_layerľ{"class_name": "LeakyReLU", "name": "leaky_re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ď
Ůtrainable_variables
Ú	variables
Űregularization_losses
Ü	keras_api
+Ť&call_and_return_all_conditional_losses
Ź__call__"ş
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ý

Ýkernel
	Ţbias
ßtrainable_variables
ŕ	variables
áregularization_losses
â	keras_api
+­&call_and_return_all_conditional_losses
Ž__call__"°	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 128]}}
Ć
ătrainable_variables
ä	variables
ĺregularization_losses
ć	keras_api
+Ż&call_and_return_all_conditional_losses
°__call__"ą
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_10", "trainable": true, "dtype": "float32"}}
ć
çtrainable_variables
č	variables
éregularization_losses
ę	keras_api
+ą&call_and_return_all_conditional_losses
˛__call__"Ń
_tf_keras_layerˇ{"class_name": "LeakyReLU", "name": "leaky_re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ý

ëkernel
	ěbias
ítrainable_variables
î	variables
ďregularization_losses
đ	keras_api
+ł&call_and_return_all_conditional_losses
´__call__"°	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 128]}}
Ć
ńtrainable_variables
ň	variables
óregularization_losses
ô	keras_api
+ľ&call_and_return_all_conditional_losses
ś__call__"ą
_tf_keras_layer{"class_name": "PixelNorm", "name": "pixel_norm_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_norm_11", "trainable": true, "dtype": "float32"}}
ć
őtrainable_variables
ö	variables
÷regularization_losses
ř	keras_api
+ˇ&call_and_return_all_conditional_losses
¸__call__"Ń
_tf_keras_layerˇ{"class_name": "LeakyReLU", "name": "leaky_re_lu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
Ů

ůkernel
	úbias
űtrainable_variables
ü	variables
ýregularization_losses
ţ	keras_api
+š&call_and_return_all_conditional_losses
ş__call__"Ź	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 128]}}

30
41
=2
>3
K4
L5
]6
^7
k8
l9
}10
~11
12
13
14
15
Ť16
Ź17
˝18
ž19
Ë20
Ě21
Ý22
Ţ23
ë24
ě25
ů26
ú27"
trackable_list_wrapper

30
41
=2
>3
K4
L5
]6
^7
k8
l9
}10
~11
12
13
14
15
Ť16
Ź17
˝18
ž19
Ë20
Ě21
Ý22
Ţ23
ë24
ě25
ů26
ú27"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
.trainable_variables
 ˙layer_regularization_losses
layers
layer_metrics
non_trainable_variables
metrics
/	variables
0regularization_losses
â__call__
á_default_save_signature
+ŕ&call_and_return_all_conditional_losses
'ŕ"call_and_return_conditional_losses"
_generic_user_object
-
ťserving_default"
signature_map
:	b2dense/kernel
:2
dense/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
5trainable_variables
 layer_regularization_losses
6	variables
layer_metrics
non_trainable_variables
metrics
layers
7regularization_losses
ä__call__
+ă&call_and_return_all_conditional_losses
'ă"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
9trainable_variables
 layer_regularization_losses
:	variables
layer_metrics
non_trainable_variables
metrics
layers
;regularization_losses
ć__call__
+ĺ&call_and_return_all_conditional_losses
'ĺ"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d/kernel
:2conv2d/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
?trainable_variables
 layer_regularization_losses
@	variables
layer_metrics
non_trainable_variables
metrics
layers
Aregularization_losses
č__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Ctrainable_variables
 layer_regularization_losses
D	variables
layer_metrics
non_trainable_variables
metrics
layers
Eregularization_losses
ę__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Gtrainable_variables
 layer_regularization_losses
H	variables
layer_metrics
non_trainable_variables
metrics
layers
Iregularization_losses
ě__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_1/kernel
:2conv2d_1/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Mtrainable_variables
 layer_regularization_losses
N	variables
layer_metrics
non_trainable_variables
 metrics
Ąlayers
Oregularization_losses
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Qtrainable_variables
 ˘layer_regularization_losses
R	variables
Łlayer_metrics
¤non_trainable_variables
Ľmetrics
Ślayers
Sregularization_losses
đ__call__
+ď&call_and_return_all_conditional_losses
'ď"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Utrainable_variables
 §layer_regularization_losses
V	variables
¨layer_metrics
Šnon_trainable_variables
Şmetrics
Ťlayers
Wregularization_losses
ň__call__
+ń&call_and_return_all_conditional_losses
'ń"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Ytrainable_variables
 Źlayer_regularization_losses
Z	variables
­layer_metrics
Žnon_trainable_variables
Żmetrics
°layers
[regularization_losses
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_3/kernel
:2conv2d_3/bias
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
_trainable_variables
 ąlayer_regularization_losses
`	variables
˛layer_metrics
łnon_trainable_variables
´metrics
ľlayers
aregularization_losses
ö__call__
+ő&call_and_return_all_conditional_losses
'ő"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
ctrainable_variables
 ślayer_regularization_losses
d	variables
ˇlayer_metrics
¸non_trainable_variables
šmetrics
şlayers
eregularization_losses
ř__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
gtrainable_variables
 ťlayer_regularization_losses
h	variables
źlayer_metrics
˝non_trainable_variables
žmetrics
żlayers
iregularization_losses
ú__call__
+ů&call_and_return_all_conditional_losses
'ů"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_4/kernel
:2conv2d_4/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
mtrainable_variables
 Ŕlayer_regularization_losses
n	variables
Álayer_metrics
Ânon_trainable_variables
Ămetrics
Älayers
oregularization_losses
ü__call__
+ű&call_and_return_all_conditional_losses
'ű"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
qtrainable_variables
 Ĺlayer_regularization_losses
r	variables
Ćlayer_metrics
Çnon_trainable_variables
Čmetrics
Élayers
sregularization_losses
ţ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
utrainable_variables
 Ęlayer_regularization_losses
v	variables
Ëlayer_metrics
Ěnon_trainable_variables
Ímetrics
Îlayers
wregularization_losses
__call__
+˙&call_and_return_all_conditional_losses
'˙"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
ytrainable_variables
 Ďlayer_regularization_losses
z	variables
Đlayer_metrics
Ńnon_trainable_variables
Ňmetrics
Ólayers
{regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_6/kernel
:2conv2d_6/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
ˇ
trainable_variables
 Ôlayer_regularization_losses
	variables
Őlayer_metrics
Önon_trainable_variables
×metrics
Řlayers
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
 Ůlayer_regularization_losses
	variables
Úlayer_metrics
Űnon_trainable_variables
Ümetrics
Ýlayers
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
 Ţlayer_regularization_losses
	variables
ßlayer_metrics
ŕnon_trainable_variables
ámetrics
âlayers
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_7/kernel
:2conv2d_7/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
 ălayer_regularization_losses
	variables
älayer_metrics
ĺnon_trainable_variables
ćmetrics
çlayers
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
 člayer_regularization_losses
	variables
élayer_metrics
ęnon_trainable_variables
ëmetrics
ělayers
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
 ílayer_regularization_losses
	variables
îlayer_metrics
ďnon_trainable_variables
đmetrics
ńlayers
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
 ňlayer_regularization_losses
	variables
ólayer_metrics
ônon_trainable_variables
őmetrics
ölayers
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_9/kernel
:2conv2d_9/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
 ÷layer_regularization_losses
 	variables
řlayer_metrics
ůnon_trainable_variables
úmetrics
űlayers
Ąregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Łtrainable_variables
 ülayer_regularization_losses
¤	variables
ýlayer_metrics
ţnon_trainable_variables
˙metrics
layers
Ľregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§trainable_variables
 layer_regularization_losses
¨	variables
layer_metrics
non_trainable_variables
metrics
layers
Šregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_10/kernel
:2conv2d_10/bias
0
Ť0
Ź1"
trackable_list_wrapper
0
Ť0
Ź1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­trainable_variables
 layer_regularization_losses
Ž	variables
layer_metrics
non_trainable_variables
metrics
layers
Żregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ątrainable_variables
 layer_regularization_losses
˛	variables
layer_metrics
non_trainable_variables
metrics
layers
łregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ľtrainable_variables
 layer_regularization_losses
ś	variables
layer_metrics
non_trainable_variables
metrics
layers
ˇregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
štrainable_variables
 layer_regularization_losses
ş	variables
layer_metrics
non_trainable_variables
metrics
layers
ťregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_12/kernel
:2conv2d_12/bias
0
˝0
ž1"
trackable_list_wrapper
0
˝0
ž1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
żtrainable_variables
 layer_regularization_losses
Ŕ	variables
layer_metrics
non_trainable_variables
metrics
layers
Áregularization_losses
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ătrainable_variables
 layer_regularization_losses
Ä	variables
 layer_metrics
Ąnon_trainable_variables
˘metrics
Łlayers
Ĺregularization_losses
˘__call__
+Ą&call_and_return_all_conditional_losses
'Ą"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çtrainable_variables
 ¤layer_regularization_losses
Č	variables
Ľlayer_metrics
Śnon_trainable_variables
§metrics
¨layers
Éregularization_losses
¤__call__
+Ł&call_and_return_all_conditional_losses
'Ł"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_13/kernel
:2conv2d_13/bias
0
Ë0
Ě1"
trackable_list_wrapper
0
Ë0
Ě1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ítrainable_variables
 Šlayer_regularization_losses
Î	variables
Şlayer_metrics
Ťnon_trainable_variables
Źmetrics
­layers
Ďregularization_losses
Ś__call__
+Ľ&call_and_return_all_conditional_losses
'Ľ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ńtrainable_variables
 Žlayer_regularization_losses
Ň	variables
Żlayer_metrics
°non_trainable_variables
ąmetrics
˛layers
Óregularization_losses
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Őtrainable_variables
 łlayer_regularization_losses
Ö	variables
´layer_metrics
ľnon_trainable_variables
śmetrics
ˇlayers
×regularization_losses
Ş__call__
+Š&call_and_return_all_conditional_losses
'Š"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ůtrainable_variables
 ¸layer_regularization_losses
Ú	variables
šlayer_metrics
şnon_trainable_variables
ťmetrics
źlayers
Űregularization_losses
Ź__call__
+Ť&call_and_return_all_conditional_losses
'Ť"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_15/kernel
:2conv2d_15/bias
0
Ý0
Ţ1"
trackable_list_wrapper
0
Ý0
Ţ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßtrainable_variables
 ˝layer_regularization_losses
ŕ	variables
žlayer_metrics
żnon_trainable_variables
Ŕmetrics
Álayers
áregularization_losses
Ž__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ătrainable_variables
 Âlayer_regularization_losses
ä	variables
Ălayer_metrics
Änon_trainable_variables
Ĺmetrics
Ćlayers
ĺregularization_losses
°__call__
+Ż&call_and_return_all_conditional_losses
'Ż"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
çtrainable_variables
 Çlayer_regularization_losses
č	variables
Člayer_metrics
Énon_trainable_variables
Ęmetrics
Ëlayers
éregularization_losses
˛__call__
+ą&call_and_return_all_conditional_losses
'ą"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_16/kernel
:2conv2d_16/bias
0
ë0
ě1"
trackable_list_wrapper
0
ë0
ě1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ítrainable_variables
 Ělayer_regularization_losses
î	variables
Ílayer_metrics
Înon_trainable_variables
Ďmetrics
Đlayers
ďregularization_losses
´__call__
+ł&call_and_return_all_conditional_losses
'ł"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ńtrainable_variables
 Ńlayer_regularization_losses
ň	variables
Ňlayer_metrics
Ónon_trainable_variables
Ômetrics
Őlayers
óregularization_losses
ś__call__
+ľ&call_and_return_all_conditional_losses
'ľ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
őtrainable_variables
 Ölayer_regularization_losses
ö	variables
×layer_metrics
Řnon_trainable_variables
Ůmetrics
Úlayers
÷regularization_losses
¸__call__
+ˇ&call_and_return_all_conditional_losses
'ˇ"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_17/kernel
:2conv2d_17/bias
0
ů0
ú1"
trackable_list_wrapper
0
ů0
ú1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
űtrainable_variables
 Űlayer_regularization_losses
ü	variables
Ülayer_metrics
Ýnon_trainable_variables
Ţmetrics
ßlayers
ýregularization_losses
ş__call__
+š&call_and_return_all_conditional_losses
'š"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ţ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ö2ó
J__inference_functional_19_layer_call_and_return_conditional_losses_6186917
J__inference_functional_19_layer_call_and_return_conditional_losses_6187901
J__inference_functional_19_layer_call_and_return_conditional_losses_6187606
J__inference_functional_19_layer_call_and_return_conditional_losses_6186813Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŕ2Ý
"__inference__wrapped_model_6185877ś
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙b
2
/__inference_functional_19_layer_call_fn_6187962
/__inference_functional_19_layer_call_fn_6187248
/__inference_functional_19_layer_call_fn_6187083
/__inference_functional_19_layer_call_fn_6188023Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ě2é
B__inference_dense_layer_call_and_return_conditional_losses_6188033˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ń2Î
'__inference_dense_layer_call_fn_6188042˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_reshape_layer_call_and_return_conditional_losses_6188056˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ó2Đ
)__inference_reshape_layer_call_fn_6188061˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_conv2d_layer_call_and_return_conditional_losses_6188071˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_conv2d_layer_call_fn_6188080˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ď2ě
G__inference_pixel_norm_layer_call_and_return_conditional_losses_6188096 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
,__inference_pixel_norm_layer_call_fn_6188101 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6188106˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
-__inference_leaky_re_lu_layer_call_fn_6188111˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ď2ě
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6188121˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_conv2d_1_layer_call_fn_6188130˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_6188146 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_1_layer_call_fn_6188151 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_6188156˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_1_layer_call_fn_6188161˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˛2Ż
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6185890ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
/__inference_up_sampling2d_layer_call_fn_6185896ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ď2ě
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6188171˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_conv2d_3_layer_call_fn_6188180˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_6188196 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_2_layer_call_fn_6188201 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_6188206˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_2_layer_call_fn_6188211˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ď2ě
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6188221˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_conv2d_4_layer_call_fn_6188230˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_6188246 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_3_layer_call_fn_6188251 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_6188256˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_3_layer_call_fn_6188261˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
´2ą
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6185909ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
1__inference_up_sampling2d_1_layer_call_fn_6185915ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ď2ě
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6188271˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_conv2d_6_layer_call_fn_6188280˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_6188296 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_4_layer_call_fn_6188301 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_6188306˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_4_layer_call_fn_6188311˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ď2ě
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6188321˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_conv2d_7_layer_call_fn_6188330˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_6188346 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_5_layer_call_fn_6188351 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_6188356˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_5_layer_call_fn_6188361˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
´2ą
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6185928ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
1__inference_up_sampling2d_2_layer_call_fn_6185934ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ď2ě
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6188371˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_conv2d_9_layer_call_fn_6188380˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_6188396 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_6_layer_call_fn_6188401 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_6188406˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_6_layer_call_fn_6188411˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6188421˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_conv2d_10_layer_call_fn_6188430˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_6188446 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_7_layer_call_fn_6188451 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_6188456˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_7_layer_call_fn_6188461˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
´2ą
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6185947ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
1__inference_up_sampling2d_3_layer_call_fn_6185953ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ2í
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6188471˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_conv2d_12_layer_call_fn_6188480˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_6188496 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_8_layer_call_fn_6188501 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_6188506˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_8_layer_call_fn_6188511˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6188521˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_conv2d_13_layer_call_fn_6188530˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ń2î
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_6188546 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
.__inference_pixel_norm_9_layer_call_fn_6188551 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ô2ń
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_6188556˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ů2Ö
/__inference_leaky_re_lu_9_layer_call_fn_6188561˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
´2ą
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6185966ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
1__inference_up_sampling2d_4_layer_call_fn_6185972ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
đ2í
F__inference_conv2d_15_layer_call_and_return_conditional_losses_6188571˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_conv2d_15_layer_call_fn_6188580˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_6188596 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
/__inference_pixel_norm_10_layer_call_fn_6188601 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ő2ň
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_6188606˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ú2×
0__inference_leaky_re_lu_10_layer_call_fn_6188611˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_conv2d_16_layer_call_and_return_conditional_losses_6188621˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_conv2d_16_layer_call_fn_6188630˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_6188646 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
/__inference_pixel_norm_11_layer_call_fn_6188651 
˛
FullArgSpec
args
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ő2ň
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_6188656˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ú2×
0__inference_leaky_re_lu_11_layer_call_fn_6188661˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_conv2d_17_layer_call_and_return_conditional_losses_6188672˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_conv2d_17_layer_call_fn_6188681˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
4B2
%__inference_signature_wrapper_6187311input_1Č
"__inference__wrapped_model_6185877Ą,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙b
Ş "?Ş<
:
	conv2d_17-*
	conv2d_17˙˙˙˙˙˙˙˙˙ß
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6188421ŤŹJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˇ
+__inference_conv2d_10_layer_call_fn_6188430ŤŹJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ß
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6188471˝žJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˇ
+__inference_conv2d_12_layer_call_fn_6188480˝žJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ß
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6188521ËĚJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˇ
+__inference_conv2d_13_layer_call_fn_6188530ËĚJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ß
F__inference_conv2d_15_layer_call_and_return_conditional_losses_6188571ÝŢJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˇ
+__inference_conv2d_15_layer_call_fn_6188580ÝŢJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ß
F__inference_conv2d_16_layer_call_and_return_conditional_losses_6188621ëěJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˇ
+__inference_conv2d_16_layer_call_fn_6188630ëěJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ţ
F__inference_conv2d_17_layer_call_and_return_conditional_losses_6188672ůúJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ś
+__inference_conv2d_17_layer_call_fn_6188681ůúJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ˇ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6188121nKL8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
*__inference_conv2d_1_layer_call_fn_6188130aKL8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙Ü
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6188171]^J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ´
*__inference_conv2d_3_layer_call_fn_6188180]^J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ü
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6188221klJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ´
*__inference_conv2d_4_layer_call_fn_6188230klJ˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ü
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6188271}~J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ´
*__inference_conv2d_6_layer_call_fn_6188280}~J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ţ
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6188321J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ś
*__inference_conv2d_7_layer_call_fn_6188330J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ţ
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6188371J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ś
*__inference_conv2d_9_layer_call_fn_6188380J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ľ
C__inference_conv2d_layer_call_and_return_conditional_losses_6188071n=>8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
(__inference_conv2d_layer_call_fn_6188080a=>8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙Ł
B__inference_dense_layer_call_and_return_conditional_losses_6188033]34/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙b
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 {
'__inference_dense_layer_call_fn_6188042P34/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙b
Ş "˙˙˙˙˙˙˙˙˙ř
J__inference_functional_19_layer_call_and_return_conditional_losses_6186813Š,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙b
p

 
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ř
J__inference_functional_19_layer_call_and_return_conditional_losses_6186917Š,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙b
p 

 
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ç
J__inference_functional_19_layer_call_and_return_conditional_losses_6187606,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙b
p

 
Ş "/˘,
%"
0˙˙˙˙˙˙˙˙˙
 ç
J__inference_functional_19_layer_call_and_return_conditional_losses_6187901,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙b
p 

 
Ş "/˘,
%"
0˙˙˙˙˙˙˙˙˙
 Đ
/__inference_functional_19_layer_call_fn_6187083,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙b
p

 
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Đ
/__inference_functional_19_layer_call_fn_6187248,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙b
p 

 
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ď
/__inference_functional_19_layer_call_fn_6187962,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙b
p

 
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ď
/__inference_functional_19_layer_call_fn_6188023,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙b
p 

 
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ţ
K__inference_leaky_re_lu_10_layer_call_and_return_conditional_losses_6188606J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ś
0__inference_leaky_re_lu_10_layer_call_fn_6188611J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ţ
K__inference_leaky_re_lu_11_layer_call_and_return_conditional_losses_6188656J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ś
0__inference_leaky_re_lu_11_layer_call_fn_6188661J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙¸
J__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_6188156j8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
/__inference_leaky_re_lu_1_layer_call_fn_6188161]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_6188206J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_2_layer_call_fn_6188211J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_6188256J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_3_layer_call_fn_6188261J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_6188306J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_4_layer_call_fn_6188311J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_6188356J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_5_layer_call_fn_6188361J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_6188406J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_6_layer_call_fn_6188411J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_6188456J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_7_layer_call_fn_6188461J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_6188506J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_8_layer_call_fn_6188511J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ý
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_6188556J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
/__inference_leaky_re_lu_9_layer_call_fn_6188561J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ś
H__inference_leaky_re_lu_layer_call_and_return_conditional_losses_6188106j8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
-__inference_leaky_re_lu_layer_call_fn_6188111]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙Ű
J__inference_pixel_norm_10_layer_call_and_return_conditional_losses_6188596H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˛
/__inference_pixel_norm_10_layer_call_fn_6188601H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ű
J__inference_pixel_norm_11_layer_call_and_return_conditional_losses_6188646H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˛
/__inference_pixel_norm_11_layer_call_fn_6188651H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ľ
I__inference_pixel_norm_1_layer_call_and_return_conditional_losses_6188146h6˘3
,˘)
'$
data˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
.__inference_pixel_norm_1_layer_call_fn_6188151[6˘3
,˘)
'$
data˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_2_layer_call_and_return_conditional_losses_6188196H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_2_layer_call_fn_6188201H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_3_layer_call_and_return_conditional_losses_6188246H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_3_layer_call_fn_6188251H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_4_layer_call_and_return_conditional_losses_6188296H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_4_layer_call_fn_6188301H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_5_layer_call_and_return_conditional_losses_6188346H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_5_layer_call_fn_6188351H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_6_layer_call_and_return_conditional_losses_6188396H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_6_layer_call_fn_6188401H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_7_layer_call_and_return_conditional_losses_6188446H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_7_layer_call_fn_6188451H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_8_layer_call_and_return_conditional_losses_6188496H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_8_layer_call_fn_6188501H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ú
I__inference_pixel_norm_9_layer_call_and_return_conditional_losses_6188546H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ą
.__inference_pixel_norm_9_layer_call_fn_6188551H˘E
>˘;
96
data,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ł
G__inference_pixel_norm_layer_call_and_return_conditional_losses_6188096h6˘3
,˘)
'$
data˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
,__inference_pixel_norm_layer_call_fn_6188101[6˘3
,˘)
'$
data˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙Ş
D__inference_reshape_layer_call_and_return_conditional_losses_6188056b0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
)__inference_reshape_layer_call_fn_6188061U0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙Ö
%__inference_signature_wrapper_6187311Ź,34=>KL]^kl}~ŤŹ˝žËĚÝŢëěůú;˘8
˘ 
1Ş.
,
input_1!
input_1˙˙˙˙˙˙˙˙˙b"?Ş<
:
	conv2d_17-*
	conv2d_17˙˙˙˙˙˙˙˙˙ď
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_6185909R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ç
1__inference_up_sampling2d_1_layer_call_fn_6185915R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ď
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_6185928R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ç
1__inference_up_sampling2d_2_layer_call_fn_6185934R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ď
L__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_6185947R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ç
1__inference_up_sampling2d_3_layer_call_fn_6185953R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ď
L__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_6185966R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ç
1__inference_up_sampling2d_4_layer_call_fn_6185972R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙í
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_6185890R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ĺ
/__inference_up_sampling2d_layer_call_fn_6185896R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙