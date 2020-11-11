
# ionosphere-machine-learning

## Sobre a base de dados

A base de dados Ionosphere contém os dados retornados de um radar, localizado na ionosfera. Os dados são coletados por um sistema localizado na base aérea de Goose Bay, localizada em Labrador no Canadá. Esse sistema consiste em uma matriz de fases de 16 antenas de alta frequência que tem como alvos os elétrons livres na ionosfera. Os retornos tidos como bons, são que aqueles que detectam evidencias que qualquer tipo de estrutura presente na ionosfera, os tidos como ruins são aqueles que não detectam. Os sinais recebidos foram processados usando uma função de autocorrelação cujos argumentos são o tempo de um pulso e o número de pulsos. As instancias dessa base de dados são descritas por 2 atributos por número de pulsos, correspondendo aos valores complexos devolvidos pela função resultante do sinal eletromagnético.

## Treinando os classificadores

### **KNN**

Rodamos o KNN não ponderado e o KNN ponderado pelo inverso da distância euclidiana, em ambos variando o K em 20 vezes. O melhor K obtido no KNN não ponderado foi 2 e alcançou a acurácia de 87% Já para o KNN ponderado pelo inverso da distância euclidiana o melhor K foi 3 com uma acurácia de 82%. Logo, podemos concluir que o melhor KNN para a base de dados foi o KNN não ponderado com um K = 2.

### **Árvore de decisão**

Primeiramente rodamos a árvore de decisão sem nenhuma poda e percebemos que nesse caso a densidade máxima da árvore foi igual a 7. Então rodamos um laço de repetição que percorria os valores de 1 até a densidade alcançada anteriormente + 1. Nesse laço, percorremos a árvore com todos os possíveis tipos de poda mais a execução sem nenhuma poda. E foi percebido que a árvore de decisão com poda obteve o melhor desempenho com uma acurácia de 97% com a densidade = 5.

### **Naive Bayes**

O Naive Bayes não possui nenhum parâmetro e ao ser executado chegou a uma acurácia de 93%.

### **SVM**

Rodamos o SVM radial e o SVM polinomial, em ambos variando o C em 20 vezes. O melhor C obtido no SVM radial foi 3 e alcançou a acurácia de 93%. Já para o SVM polinomial o melhor C foi 1 com uma acurácia de 90%. Logo, podemos concluir que o melhor SVM para a base de dados foi SVM radial com um C = 3 e acurácia de 93%.

###  **MLP**

No MLP os parâmetros foram: Número de épocas de treino, taxa de aprendizagem e número de camadas escondidas. O número de camadas escondidas foi variado incrementando de 50 em 50  a cada repetição até chegar em 500 , a taxa de aprendizagem foi variada de incrementando 0.1 a cada repetição até chegar em 1 e número de épocas de treino foi variado incrementando de 200 em 200 a cada repetição até chegar em 2000. No final de todas as repetições o classificador percorreu um total de 1000 combinações. Os melhores parâmetros foram: Número de épocas de treino = 1400; Taxa de aprendizagem = 0.9; Número de camadas escondidas = 400 com uma acurácia de 94%.

### Tabela com as acurácias
|  				REPETIÇÃO 			  |  				KNN 			             |  				AD 			              |  				NB 			                |  				SVM 			               |  				MLP 			      |
|:------------:|:-----------------:|:-----------------:|:-------------------:|:-------------------:|:----------:|
|       1        |  				0.909090 			        |  				0.943181 			        |  				0.806818 			          |  				0.954545 			          |  				0.943181 			 |
|  				2 			      |  				0.875 			           |  				0.909091 			        |  				0.897727 			          |  				0.909091 			          |  				0.852272 			 |
|  				3 			      |  				0.863636 			        |  				0.863636 			        |  				0.931818 			          |  				0.920454 			          |  				0.886363 			 |
|  				4 			      |  				0.897727 			        |  				0.909091 			        |  				0.954545 			          |  				0.943182 			          |  				0.818181 			 |
|  				5 			      |  				0.897727 			        |  				0.806818 			        |  				0.886364 			          |  				0.920454 			          |  				0.852272 			 |
|  				6 			      |  				0.852278 			        |  				0.886364 			        |  				0.840909 			          |  				0.943182 			          |  				0.818181 			 |
|  				7 			     |  				0.875 			           |  				0.875 			           |  				0.852273 			          |  				0.897727 			          |  				0.852272 			 |
|  				8 			      |  				0.897727 			        |  				0.840909 			        |  				0.909091 			          |  				0.954545 			          |  				0.875 			    |
|  				9 			      |  				0.852278 			        |  				0.909091 			        |  				0.795454 			          |  				0.920454 			          |  				0.875 			    |
|  				10 			     |  				0.909091 			        |  				0.852273 			        |  				0.875 			             |  				0.954545 			          |  				0.875 			    |
|  				11 			     |  				0.909091 			        |  				0.909091 			        |  				0.943182 			          |  				0.943182 			          |  				0.818181 			 |
|  				12 			     |  				0.931818 			        |  				0.920454 			        |  				0.886364 			          |  				0.965909 			          |  				0.886363 			 |
|  				13 			    |  				0.886364 			        |  				0.863636 			        |  				0.897727 			          |  				0.954545 			          |  				0.897727 			 |
|  				14 			     |  				0.863636 			        |  				0.886364 			        |  				0.829545 			          |  				0.931818 			          |  				0.897727 			 |
|  				15 			     |  				0.897727 			        |  				0.829545 			        |  				0.920454 			          |  				0.954545 			          |  				0.863636 			 |
|  				16 			     |  				0.920454 			        |  				0.886364 			        |  				0.863636 			          |  				0.931818 			          |  				0.806818 			 |
|  				17 			     |  				0.863636 			        |  				0.931818 			        |  				0.931818 			          |  				0.943182 			          |  				0.829545 			 |
|  				18 			     |  				0.886364 			        |  				0.886363 			        |  				0.920454 			          |  				0.954545 			          |  				0.829545 			 |
|  				19 			     |  				0.897727 			        |  				0.909091 			        |  				0.909091 			          |  				0.920454 			          |  				0.875 			    |
|  				20 			    |  				0.886364 			        |  				0.829545 			        |  				0.886363 			          |  				0.954545 			          |  				0.943181 			 |
|  			**MÉDIA (DP)** 			 |  			**0.888636 (0.02)** 			 |  			**0.882386 (0.03)** 			 |  			**0.886932 (0.04)** 			 |  			**0.938636 (0.02)** 			 | **0.737499 (0.03)** |


​	Tabela contendo as acurácias obtidas após rodar todos os classificadores com os melhores parâmetros para o conjunto de validação.


​	Após a montagem da tabela, contendo todas as acurácias, foram feitos dois testes estatísticos. O primeiro, usando o Teste de Kruskal-Wallis com nível de confiança em 95%, foi feito para identificar se existe alguma diferença no desempenho dos algoritmos (independentemente de qual foi melhor ou pior).

<img src="file:///tmp/lu60957cneqwv.tmp/lu60957cneqx8_tmp_d8c75df3a570a22c.png" alt="img" style="zoom:100%;" />

O teste de Kruskal-Wallis mostrou que existe uma significativa diferença entre os classificadores. Rejeitando a hipótese Ho e confirmando que existe um classificador melhor. Após descobrir que existe um classificador melhor, foi feito o teste de Mann-Whitney com nível de confiança de 95% para descobrir qual o classificador com o melhor desempenho.

Para esse teste os classificadores são comparados dois a dois.

#### KNN x Árvore de decisão

|  					REPETIÇÃO 				  |  					KNN 				             |  					AD 				              |
|:------------:|:-----------------:|:-----------------:|
|  					1 				          |  					0.909090 				        |  					0.943181 				        |
|  					2 				          |  					0.875 				           |  					0.909091 				        |
|  					3 				          |  					0.863636 				        |  					0.863636 				        |
|  					4 				          |  					0.897727 				        |  					0.909091 				        |
|  					5 				          |  					0.897727 				        |  					0.806818 				        |
|  					6 				          |  					0.852278 				        |  					0.886364 				        |
|  					7 				          |  					0.875 				           |  					0.875 				           |
|  					8 				          |  					0.897727 				        |  					0.840909 				        |
|  					9 				          |  					0.852278 				        |  					0.909091 				        |
|  					10 				         |  					0.909091 				        |  					0.852273 				        |
|  					11 				         |  					0.909091 				        |  					0.909091 				        |
|  					12 				         |  					0.931818 				        |  					0.920454 				        |
|  					13 				         |  					0.886364 				        |  					0.863636 				        |
|  					14 				         |  					0.863636 				        |  					0.886364 				        |
|  					15 				         |  					0.897727 				        |  					0.829545 				        |
|  					16 				         |  					0.920454 				        |  					0.886364 				        |
|  					17 				         |  					0.863636 				        |  					0.931818 				        |
|  					18 				         |  					0.886364 				        |  					0.886363 				        |
|  					19 				         |  					0.897727 				        |  					0.909091 				        |
|  					20 				         |  					0.886364 				        |  					0.829545 				        |
|  			**MÉDIA (DP)** 			|  			**0.888636 (0.02)** 			|  			**0.882386 (0.03)** 			|

<img src="file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_33dc6090b22e181f.png" alt="img" style="zoom:100%;" />

A diferença entre os classificadores não foi significativa. Portanto, aceita-se Ho.

#### KNN x Naive Bayes

|  					REPETIÇÃO 				  |  					KNN 				             |  					NB 				                |
|:------------:|:-----------------:|:-------------------:|
|  					1 				          |  					0.909090 				        |  					0.806818 				          |
|  					2 				          |  					0.875 				           |  					0.897727 				          |
|  					3 				          |  					0.863636 				        |  					0.931818 				          |
|  					4 				          |  					0.897727 				        |  					0.954545 				          |
|  					5 				          |  					0.897727 				        |  					0.886364 				          |
|  					6 				          |  					0.852278 				        |  					0.840909 				          |
|  					7 				          |  					0.875 				           |  					0.852273 				          |
|  					8 				          |  					0.897727 				        |  					0.909091 				          |
|  					9 				          |  					0.852278 				        |  					0.795454 				          |
|  					10 				         |  					0.909091 				        |  					0.875 				             |
|  					11 				         |  					0.909091 				        |  					0.943182 				          |
|  					12 				         |  					0.931818 				        |  					0.886364 				          |
|  					13 				         |  					0.886364 				        |  					0.897727 				          |
|  					14 				         |  					0.863636 				        |  					0.829545 				          |
|  					15 				         |  					0.897727 				        |  					0.920454 				          |
|  					16 				         |  					0.920454 				        |  					0.863636 				          |
|  					17 				         |  					0.863636 				        |  					0.931818 				          |
|  					18 				         |  					0.886364 				        |  					0.920454 				          |
|  					19 				         |  					0.897727 				        |  					0.909091 				          |
|  					20 				         |  					0.886364 				        |  					0.886363 				          |
|  			**MÉDIA (DP)** 			|  			**0.888636 (0.02)** 			|  			**0.886932 (0.04)** 			|

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_94dfd5ca28d0fd8f.png)

A diferença entre os classificadores não foi significativa. Portanto, aceita-se Ho.

#### KNN x SVM

|  					REPETIÇÃO 				  |  					KNN 				             |  					SVM 				               |
|:------------:|:-----------------:|:-------------------:|
|  					1 				          |  					0.909090 				        |  					0.954545 				          |
|  					2 				          |  					0.875 				           |  					0.909091 				          |
|  					3 				          |  					0.863636 				        |  					0.920454 				          |
|  					4 				          |  					0.897727 				        |  					0.943182 				          |
|  					5 				          |  					0.897727 				        |  					0.920454 				          |
|  					6 				          |  					0.852278 				        |  					0.943182 				          |
|  					7 				          |  					0.875 				           |  					0.897727 				          |
|  					8 				          |  					0.897727 				        |  					0.954545 				          |
|  					9 				          |  					0.852278 				        |  					0.920454 				          |
|  					10 				         |  					0.909091 				        |  					0.954545 				          |
|  					11 				         |  					0.909091 				        |  					0.943182 				          |
|  					12 				         |  					0.931818 				        |  					0.965909 				          |
|  					13 				         |  					0.886364 				        |  					0.954545 				          |
|  					14 				         |  					0.863636 				        |  					0.931818 				          |
|  					15 				         |  					0.897727 				        |  					0.954545 				          |
|  					16 				         |  					0.920454 				        |  					0.931818 				          |
|  					17 				         |  					0.863636 				        |  					0.943182 				          |
|  					18 				         |  					0.886364 				        |  					0.954545 				          |
|  					19 				         |  					0.897727 				        |  					0.920454 				          |
|  					20 				         |  					0.886364 				        |  					0.954545 				          |
|  			**MÉDIA (DP)** 			|  			**0.888636 (0.02)** 			|  			**0.938636 (0.02)** 			|

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_8bfc707ea378bfd4.png)

A diferença entre os classificadores é significativa. Portanto, rejeita-se Ho

#### KNN x MLP

|  					REPETIÇÃO 				  |  					KNN 				             |  					MLP 				               |
|:------------:|:-----------------:|:-------------------:|
|  					1 				          |  					0.909090 				        |  					0.943181 				          |
|  					2 				          |  					0.875 				           |  					0.852272 				          |
|  					3 				          |  					0.863636 				        |  					0.886363 				          |
|  					4 				          |  					0.897727 				        |  					0.818181 				          |
|  					5 				          |  					0.897727 				        |  					0.852272 				          |
|  					6 				          |  					0.852278 				        |  					0.818181 				          |
|  					7 				          |  					0.875 				           |  					0.852272 				          |
|  					8 				          |  					0.897727 				        |  					0.875 				             |
|  					9 				          |  					0.852278 				        |  					0.875 				             |
|  					10 				         |  					0.909091 				        |  					0.875 				             |
|  					11 				         |  					0.909091 				        |  					0.818181 				          |
|  					12 				         |  					0.931818 				        |  					0.886363 				          |
|  					13 				         |  					0.886364 				        |  					0.897727 				          |
|  					14 				         |  					0.863636 				        |  					0.897727 				          |
|  					15 				         |  					0.897727 				        |  					0.863636 				          |
|  					16 				         |  					0.920454 				        |  					0.806818 				          |
|  					17 				         |  					0.863636 				        |  					0.829545 				          |
|  					18 				         |  					0.886364 				        |  					0.829545 				          |
|  					19 				         |  					0.897727 				        |  					0.875 				             |
|  					20 				         |  					0.886364 				        |  					0.943181 				          |
|  			**MÉDIA (DP)** 			|  			**0.888636 (0.02)** 			|  			**0.737499 (0.03)** 			|
![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_151e8c535470463.png)

A diferença entre os classificadores é significativa. Portanto, rejeita-se Ho

#### Árvore de decisão x Naive Bayes

|  					REPETIÇÃO 				  |  					AD 				              |  					NB 				                |
|:------------:|:-----------------:|:-------------------:|
|  					1 				          |  					0.943181 				        |  					0.806818 				          |
|  					2 				          |  					0.909091 				        |  					0.897727 				          |
|  					3 				          |  					0.863636 				        |  					0.931818 				          |
|  					4 				          |  					0.909091 				        |  					0.954545 				          |
|  					5 				          |  					0.806818 				        |  					0.886364 				          |
|  					6 				          |  					0.886364 				        |  					0.840909 				          |
|  					7 				          |  					0.875 				           |  					0.852273 				          |
|  					8 				          |  					0.840909 				        |  					0.909091 				          |
|  					9 				          |  					0.909091 				        |  					0.795454 				          |
|  					10 				         |  					0.852273 				        |  					0.875 				             |
|  					11 				         |  					0.909091 				        |  					0.943182 				          |
|  					12 				         |  					0.920454 				        |  					0.886364 				          |
|  					13 				         |  					0.863636 				        |  					0.897727 				          |
|  					14 				         |  					0.886364 				        |  					0.829545 				          |
|  					15 				         |  					0.829545 				        |  					0.920454 				          |
|  					16 				         |  					0.886364 				        |  					0.863636 				          |
|  					17 				         |  					0.931818 				        |  					0.931818 				          |
|  					18 				         |  					0.886363 				        |  					0.920454 				          |
|  					19 				         |  					0.909091 				        |  					0.909091 				          |
|  					20 				         |  					0.829545 				        |  					0.886363 				          |
|  				**MÉDIA (DP)** 				 |  			**0.882386 (0.03)** 			|  				**0.886932 (0.04)** 				 |

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_525389dcaaccd68f.png)

A diferença entre os classificadores não foi significativa. Portanto, aceita-se Ho.

#### Árvore de decisão x SVM

|  					REPETIÇÃO 				  |  					AD 				              |  					SVM 				               |
|:------------:|:-----------------:|:-------------------:|
|  					1 				          |  					0.943181 				        |  					0.954545 				          |
|  					2 				          |  					0.909091 				        |  					0.909091 				          |
|  					3 				          |  					0.863636 				        |  					0.920454 				          |
|  					4 				          |  					0.909091 				        |  					0.943182 				          |
|  					5 				          |  					0.806818 				        |  					0.920454 				          |
|  					6 				          |  					0.886364 				        |  					0.943182 				          |
|  					7 				          |  					0.875 				           |  					0.897727 				          |
|  					8 				          |  					0.840909 				        |  					0.954545 				          |
|  					9 				          |  					0.909091 				        |  					0.920454 				          |
|  					10 				         |  					0.852273 				        |  					0.954545 				          |
|  					11 				         |  					0.909091 				        |  					0.943182 				          |
|  					12 				         |  					0.920454 				        |  					0.965909 				          |
|  					13 				         |  					0.863636 				        |  					0.954545 				          |
|  					14 				         |  					0.886364 				        |  					0.931818 				          |
|  					15 				         |  					0.829545 				        |  					0.954545 				          |
|  					16 				         |  					0.886364 				        |  					0.931818 				          |
|  					17 				         |  					0.931818 				        |  					0.943182 				          |
|  					18 				         |  					0.886363 				        |  					0.954545 				          |
|  					19 				         |  					0.909091 				        |  					0.920454 				          |
|  					20 				         |  					0.829545 				        |  					0.954545 				          |
|  			**MÉDIA (DP)** 			|  			**0.882386 (0.03)** 			|  			**0.938636 (0.02)** 			|

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_d00f776b4fd19a32.png)

A diferença entre os classificadores é significativa. Portanto, rejeita-se Ho.

#### Árvore de decisão x MLP

|  					REPETIÇÃO 				  |  					AD 				             |  					MLP 				               |
|:------------:|:-----------------:|:-------------------:|
|  					1 				          |  					0.909090 				        |  					0.943181 				          |
|  					2 				          |  					0.875 				           |  					0.852272 				          |
|  					3 				          |  					0.863636 				        |  					0.886363 				          |
|  					4 				          |  					0.897727 				        |  					0.818181 				          |
|  					5 				          |  					0.897727 				        |  					0.852272 				          |
|  					6 				          |  					0.852278 				        |  					0.818181 				          |
|  					7 				          |  					0.875 				           |  					0.852272 				          |
|  					8 				          |  					0.897727 				        |  					0.875 				             |
|  					9 				          |  					0.852278 				        |  					0.875 				             |
|  					10 				         |  					0.909091 				        |  					0.875 				             |
|  					11 				         |  					0.909091 				        |  					0.818181 				          |
|  					12 				         |  					0.931818 				        |  					0.886363 				          |
|  					13 				         |  					0.886364 				        |  					0.897727 				          |
|  					14 				         |  					0.863636 				        |  					0.897727 				          |
|  					15 				         |  					0.897727 				        |  					0.863636 				          |
|  					16 				         |  					0.920454 				        |  					0.806818 				          |
|  					17 				         |  					0.863636 				        |  					0.829545 				          |
|  					18 				         |  					0.886364 				        |  					0.829545 				          |
|  					19 				         |  					0.897727 				        |  					0.875 				             |
|  					20 				         |  					0.886364 				        |  					0.943181 				          |
|  			**MÉDIA (DP)** 			|  			**0.888636 (0.02)** 			|  			**0.737499 (0.03)** 				|

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_c95b0358e863308c.png)

A diferença entre os classificadores é significativa. Portanto, rejeita-se Ho.

#### Naive Bayes x SVM

|  					REPETIÇÃO 				  |  					NB 				                |  					SVM 				               |
|:------------:|:-------------------:|:-------------------:|
|  					1 				          |  					0.806818 				          |  					0.954545 				          |
|  					2 				          |  					0.897727 				          |  					0.909091 				          |
|  					3 				          |  					0.931818 				          |  					0.920454 				          |
|  					4 				          |  					0.954545 				          |  					0.943182 				          |
|  					5 				          |  					0.886364 				          |  					0.920454 				          |
|  					6 				          |  					0.840909 				          |  					0.943182 				          |
|  					7 				          |  					0.852273 				          |  					0.897727 				          |
|  					8 				          |  					0.909091 				          |  					0.954545 				          |
|  					9 				          |  					0.795454 				          |  					0.920454 				          |
|  					10 				         |  					0.875 				             |  					0.954545 				          |
|  					11 				         |  					0.943182 				          |  					0.943182 				          |
|  					12 				         |  					0.886364 				          |  					0.965909 				          |
|  					13 				         |  					0.897727 				          |  					0.954545 				          |
|  					14 				         |  					0.829545 				          |  					0.931818 				          |
|  					15 				         |  					0.920454 				          |  					0.954545 				          |
|  					16 				         |  					0.863636 				          |  					0.931818 				          |
|  					17 				         |  					0.931818 				          |  					0.943182 				          |
|  					18 				         |  					0.920454 				          |  					0.954545 				          |
|  					19 				         |  					0.909091 				          |  					0.920454 				          |
|  					20 				         |  					0.886363 				          |  					0.954545 				          |
|  			**MÉDIA (DP)** 			|  			**0.886932(0.04)** 			|  			**0.938636 (0.02)** 			|

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_a3921f638e34f10c.png)

A diferença entre os classificadores é significativa. Portanto, rejeita-se Ho.

#### Naive Bayes x MLP

|  					REPETIÇÃO 				  |  					NB 				                |  					MLP 				               |
|:------------:|:-------------------:|:-------------------:|
|  					1 				          |  					0.806818 				          |  					0.943181 				          |
|  					2 				          |  					0.897727 				          |  					0.852272 				          |
|  					3 				          |  					0.931818 				          |  					0.886363 				          |
|  					4 				          |  					0.954545 				          |  					0.818181 				          |
|  					5 				          |  					0.886364 				          |  					0.852272 				          |
|  					6 				          |  					0.840909 				          |  					0.818181 				          |
|  					7 				          |  					0.852273 				          |  					0.852272 				          |
|  					8 				          |  					0.909091 				          |  					0.875 				             |
|  					9 				          |  					0.795454 				          |  					0.875 				             |
|  					10 				         |  					0.875 				             |  					0.875 				             |
|  					11 				         |  					0.943182 				          |  					0.818181 				          |
|  					12 				         |  					0.886364 				          |  					0.886363 				          |
|  					13 				         |  					0.897727 				          |  					0.897727 				          |
|  					14 				         |  					0.829545 				          |  					0.897727 				          |
|  					15 				         |  					0.920454 				          |  					0.863636 				          |
|  					16 				         |  					0.863636 				          |  					0.806818 				          |
|  					17 				         |  					0.931818 				          |  					0.829545 				          |
|  					18 				         |  					0.920454 				          |  					0.829545 				          |
|  					19 				         |  					0.909091 				          |  					0.875 				             |
|  					20 				         |  					0.886363 				          |  					0.943181 				          |
|  			**MÉDIA (DP)** 			|  			**0.886932 (0.04)** 			|  			**0.737499 (0.03)** 			|

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_e5fa3ecf68e41bec.png)

A diferença entre os classificadores é significativa. Portanto, rejeita-se Ho.

#### SVM X MLP

|  					REPETIÇÃO 				  |  					SVM 				               |  					MLP 				               |
|:------------:|:-------------------:|:-------------------:|
|  					1 				          |  					0.954545 				          |  					0.943181 				          |
|  					2 				          |  					0.909091 				          |  					0.852272 				          |
|  					3 				          |  					0.920454 				          |  					0.886363 				          |
|  					4 				          |  					0.943182 				          |  					0.818181 				          |
|  					5 				          |  					0.920454 				          |  					0.852272 				          |
|  					6 				          |  					0.943182 				          |  					0.818181 				          |
|  					7 				          |  					0.897727 				          |  					0.852272 				          |
|  					8 				          |  					0.954545 				          |  					0.875 				             |
|  					9 				          |  					0.920454 				          |  					0.875 				             |
|  					10 				         |  					0.954545 				          |  					0.875 				             |
|  					11 				         |  					0.943182 				          |  					0.818181 				          |
|  					12 				         |  					0.965909 				          |  					0.886363 				          |
|  					13 				         |  					0.954545 				          |  					0.897727 				          |
|  					14 				         |  					0.931818 				          |  					0.897727 				          |
|  					15 				         |  					0.954545 				          |  					0.863636 				          |
|  					16 				         |  					0.931818 				          |  					0.806818 				          |
|  					17 				         |  					0.943182 				          |  					0.829545 				          |
|  					18 				         |  					0.954545 				          |  					0.829545 				          |
|  					19 				         |  					0.920454 				          |  					0.875 				             |
|  					20 				         |  					0.954545 				          |  					0.943181 				          |
|  			**MÉDIA (DP)** 			|  			**0.938636 (0.02)** 			|  			**0.737499 (0.03)** 			|

![img](file:///tmp/lu60957cneqwv.tmp/lu60957cneqxu_tmp_be540f020824d9b6.png)

A diferença entre os classificadores é significativa. Portanto, rejeita-se Ho.

### Conclusão

Após comparar, dois a dois todos os classificadores, chegou-se na conclusão de que: **SVM > KNN = Árvore de decisão = Naive Bayes > MLP**
