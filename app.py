import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def random_walk(coefficients, delta_coefficients):
    return coefficients + delta_coefficients

def population_model(P, t, r, K, a, b, delta_a, delta_b):
    # Update coefficients using random walks
    index = int(t)  # Convert t to an integer index
    a = random_walk(a, delta_a[:, index])
    b = random_walk(b, delta_b[:, index])
    
    # Calculate population change
    dPdt = r * P * (1 - P / K) + np.sum(a * P - b * P)
    return dPdt

continent = {
    "Africa":   13930,
    "Americas":   10310,
    "Asia":   46940,
    "Europe":   7450,
    "Oceania":   440
}
st.title('Population Dynamics: Modeling Uncertainty OF External Factors')

# Add 'Custom' option to the list of continents
continent_options = ['Custom'] + list(continent.keys())
st.subheader('Select Continent or enter a custom input')
selected_continent = st.selectbox('', continent_options)

# If 'Custom' is selected, allow the user to input a custom initial population
custom_population = None
if selected_continent == 'Custom':
    custom_population = st.slider('Enter Custom Population(in 100k)', 1, 99999, 1, 300)
else:
    # Otherwise, use the population of the selected continent as the initial population
    P0 = continent[selected_continent]
st.subheader('Intrinsic Growth Rate(number of births minus the number of deaths)')
r = st.slider('',  0.0,  1.0,  0.1,  0.01)
st.subheader('Carrying Capacity(Maximum population size of the species that the environment can sustain indefinitely)')
K = st.slider('', 500,  20000,  1000,  500)
st.subheader('Positive Factors(% of population affected by the factor)')
immigration = st.slider('Immigration',  0.0,  1.0,  0.0,  0.01)
healthcare = st.slider('Healthcare advancement',  0.0,  1.0,  0.0,  0.01)
st.subheader('Negative Factors(% of population affected by the factor)')
emmigration = st.slider('Emigration',  0.0,  1.0,  0.0,  0.01)
epidemics = st.slider('Epidemic ',  0.0,  1.0,  0.0,  0.01)

if st.button('Predict'):
    # Initial coefficients
    a_init = np.array([immigration, healthcare])
    b_init = np.array([emmigration, epidemics])

    # Use custom population if 'Custom' was selected, otherwise use the continent population
    P0 = custom_population if custom_population else continent[selected_continent]

    # Time values
    t = np.linspace(0,  100, P0)

    # Random increments for coefficients
    delta_a = np.random.normal(0, 0.002, size=(len(a_init), len(t)))
    delta_b = np.random.normal(0, 0.002, size=(len(b_init), len(t)))

    # Solve the differential equation
    solution = odeint(population_model, P0, t, args=(r, K, a_init, b_init, delta_a, delta_b))

    # Plotting the population over time
    fig, ax = plt.subplots()
    ax.plot(t, solution, label='Population')
    ax.set_xlabel('Time (in years)')
    ax.set_ylabel('Population (per million)')
    ax.set_title('Population growth model with  Coefficient Changes for {}'.format(selected_continent if not custom_population else 'Custom input'))
    ax.legend()
    st.pyplot(fig)
