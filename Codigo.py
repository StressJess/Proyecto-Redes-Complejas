import networkx as nx
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from re import T

#Se carga el archivo CSV en un DataFrame de pandas df
try:
  df = pd.read_csv('influence_data.csv')
except FileNotFoundError:
  print("Error: 'influence_data.csv' no se encuentra en el directorio actual.")
except pd.errors.ParserError:
    print("Error: No se pudo analizar el archivo CSV. Verifique su formato.")
except Exception as e:
    print(f"Un error inesperado a ocurrido: {e}")

#Se crean las digraficas que se utilizaran para el analisis de los datos
G = nx.DiGraph()
for index, row in df.iterrows():
    G.add_edge(row['influencer_name'], row['follower_name'])

G_2 = nx.DiGraph()
for index, row in df.iterrows():
    G_2.add_edge(row['influencer_id'], row['follower_id'])

G_Genre = nx.DiGraph()
for index, row in df.iterrows():
    G_Genre.add_edge(row['influencer_main_genre'], row['follower_main_genre'])

#Verificar que esten todos los datos, debe tener 5568 nodos (artistas) y 42761 conexiones
print("El orden de la gráfica es:", G.order())
print("El tamaño de la gráfica es:", G.size())

#---------------Ex-grado e In-grado------------------------------------------------------------------------------

# Grado de entrada
InDeg = dict(G_2.in_degree())
OutDeg = dict(G_2.out_degree())

InDeg_df = pd.DataFrame.from_dict(InDeg, orient='index', columns=['Grado de Entrada'])
OutDeg_df = pd.DataFrame.from_dict(OutDeg, orient='index', columns=['Grado de Salida'])

InDeg_df = InDeg_df.reset_index().rename(columns={'index': 'id'})
OutDeg_df = OutDeg_df.reset_index().rename(columns={'index': 'id'})

plt.figure(figsize=(14, 10))
ax = plt.gca()
sns.scatterplot(data=InDeg_df, x='id', y='Grado de Entrada', color='green')
plt.title("Distribucion el In-Grado")
plt.xlabel("Id Artistas")
plt.ylabel("In-Grado")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
ax = plt.gca()
sns.scatterplot(data=OutDeg_df, x='id', y='Grado de Salida', color='red')
plt.title("Distribucion el Ex-Grado")
plt.xlabel("Id Artistas")
plt.ylabel("Ex-Grado")
plt.tight_layout()
plt.show()

#------------------------Centralidades-----------------------------------------------------------------------

#Guardar en variables los diccionarios con las centralidades de cada nodo
CI = nx.betweenness_centrality(G)
DC = nx.degree_centrality(G)
EC = nx.eigenvector_centrality(G)

TopDC = sorted(DC.items(), key=lambda x: x[1], reverse=True)[:20]
nodes, centralities = zip(*TopDC)

TopCI = sorted(CI.items(), key=lambda x: x[1], reverse=True)[:20]
nodes2, centralities2 = zip(*TopCI)

TopEC = sorted(EC.items(), key=lambda x: x[1], reverse=True)[:20]
nodes4, centralities4 = zip(*TopEC)

# Crear el gráfico de barras horizontal
plt.figure(figsize=(10, 8))
plt.barh(nodes, centralities, color='blue')
plt.xlabel('Centralidad de Grado')
plt.ylabel('Artistas/Bandas')
plt.title('Top 20 Nodos por Centralidad de Grado')
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10, 8))
plt.barh(nodes2, centralities2, color='green')
plt.xlabel('Centralidad Intermedia')
plt.ylabel('Artistas/Bandas')
plt.title('Top 20 Nodos por Centralidad Intermedia')
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10, 8))
plt.barh(nodes4, centralities4, color='orange')
plt.xlabel('Centralidad de Eigenvector')
plt.ylabel('Artistas/Bandas')
plt.title('Top 20 Nodos de Eigenvector')
plt.gca().invert_yaxis()
plt.show()

#Centralidades de cada Género
CI2 = nx.betweenness_centrality(G_Genre)
DC2 = nx.degree_centrality(G_Genre)
EC2 = nx.eigenvector_centrality(G_Genre)

#Convertir el diccionario en un DataFrame para mejor manejo de los datos
DC2_df = pd.DataFrame.from_dict(DC2, orient='index', columns=['Grado'])
CI2_df = pd.DataFrame.from_dict(CI2, orient='index', columns=['Intermedia'])
EC2_df = pd.DataFrame.from_dict(EC2, orient='index', columns=['De Eigenvector'])

#Reset de los indices para tener 'Nodo' y 'Grado' como columnas
DC2_df = DC2_df.reset_index().rename(columns={'index': 'Nodo'})

plt.figure(figsize=(14, 10))
ax = plt.gca()
sns.barplot(data=DC2_df, x='Nodo', y='Grado', palette="hsv")
plt.title("Centralidad de Grado por Género")
plt.xlabel("Géneros")
plt.ylabel("Centralidad de Grado")
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()


#Concatenar los DataFrames
Result_df = pd.concat([CI2_df, EC2_df], axis=1)

# Convertir a un diccionario otra vez
Result_dict = Result_df.to_dict('index')

# Convertir el dicionario concatenado a un DataFrame
Result_df = pd.DataFrame.from_dict(Result_dict, orient='index')

# Para ayudar a seaborn
ayudita = pd.melt(Result_df.reset_index(), id_vars=['index'], var_name='Centralidad', value_name='Value')

# Create the plot
plt.figure(figsize=(8, 6)) 
sns.barplot(x='index', y='Value', hue='Centralidad', data=ayudita, palette="hsv")
plt.title('Centralidades de los Géneros')
plt.xlabel('Genero')
plt.ylabel('Valor')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()  
plt.show()

#--------------------------Popularidad--------------------------------------------------------------------------------

#Se carga el archivo CSV en un DataFrame de pandas df
try:
  df = pd.read_csv('Spotify 2010 - 2019 Top 100.csv')
except FileNotFoundError:
  print("Error: 'Spotify 2010 - 2019 Top 100.csv' no se encuentra en el directorio actual.")
except pd.errors.ParserError:
    print("Error: No se pudo analizar el archivo CSV. Verifique su formato.")
except Exception as e:
    print(f"Un error inesperado a ocurrido: {e}")


# Agrupar datos por año y género, obteniendo la popularidad promedio
heatmap_data = dfpop.groupby(['top year', 'top genre'])['pop'].mean().unstack()

# Graficar el mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="jet", fmt=".1f", linecolor='grey', linewidths=0.05)
plt.title('Mapa de Calor de Popularidad por Género y Año')
plt.xlabel('Género')
plt.ylabel('Año')
plt.show()

df = dfpop.groupby(['top year', 'top genre'])['pop'].mean().unstack()

# Graficar
df.plot(figsize=(10, 6), cmap = 'gist_ncar')
plt.xlabel('Año')
plt.ylabel('Popularidad Promedio')
plt.title('Tendencias de Popularidad por Género a lo Largo de los Años')
plt.legend(title='Género', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.show()
