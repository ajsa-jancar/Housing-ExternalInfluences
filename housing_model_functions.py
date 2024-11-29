from scipy.spatial import cKDTree
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import distance

def model_conversions(population, income):
    """
    Convert population and income data for analysis.

    Parameters:
    population (DataFrame): DataFrame where the first column contains boroughs 
                            and the second column contains actual population numbers.
    income (DataFrame): DataFrame where the first column contains floor ranges of income 
                        and the second column contains the number of people.

    Returns:
    tuple: A tuple containing two converted DataFrames:
           - Population DataFrame with proportions instead of actual numbers.
           - Income DataFrame with the first column as average income values and second column as population proportion.
    """
    
    population = population.copy()
    income = income.copy()
    
    # Convert population numbers to percentages
    total_population = population.iloc[:, 0].sum()  # Sum total population
    population.iloc[:, 0] = (population.iloc[:, 0] / total_population)  # Convert to proportion

   # Convert income floor ranges to average values
   
   # Save last row value
    last = income.iloc[-1, 0]
    
    # Calculate average prices
    income.iloc[:, 0] = (income.iloc[:, 0] + income.iloc[:, 0].shift(-1)) / 2 
    
    # Treat last row separately
    income.iloc[-1, 0] = (last + 1000000) / 2  # For last row, average with 1000000
    
    # Relabel column
    income = income.rename(columns={income.columns[0]: 'Income Average'})
    
    # Convert the second column (number of people) to percentage
    total_people = income.iloc[:, 1].sum()  # Sum total number of people
    income.iloc[:, 1] = (income.iloc[:, 1] / total_people) # Convert to proportion
    
    return population, income

def initial_price_dist(agents_df, year, price_data, type = 'median'):
    '''
    Assign an initial house price to agents based on the 'District' column in agents_df.
    
    Parameters:
    agents_df (DataFrame): DataFrame with columns 'x_coord', 'y_coord', 'District', and 'price'.
    year (int): Year of data to be considered.
    price_data (DataFrame): DataFrame containing yearly price information by district.
    type (str): Type of price assignment - 'mean', 'median', or 'actual'.
    
    Returns:
    DataFrame: Updated agents_df with assigned 'price'.
    '''
    if type not in ['mean', 'median', 'actual']:
        raise ValueError("Invalid type. Use 'mean', 'median', or 'actual'.")
    
    agents_df = agents_df.copy()
    
    # Filter price data for the selected year
    year_prices = price_data[price_data.index == year]

    if type == 'mean':
        agents_df['price'] = np.mean(year_prices)
    
    elif type == 'median':
        agents_df['price'] = np.median(year_prices)
    
    else:
        year_prices = year_prices.to_dict()
        agents_df['price'] = agents_df['District'].map(year_prices)
    
    return agents_df

def initial_affluence_dist(n, a, p, agents_df):
    '''
    Assign random affluence to each agent based on provided affluence groups and proportions
    Inputs:
    n (int): number of agents
    a (array): affluence group parameters 
    p (array): proportions of affluence groups
    agents_df (DataFrame): DataFrame with columns 'x_coord', 'y_coord', and 'affluence'
    '''
    agents_df = agents_df.copy()
    
    # Assign a random affluence to each agent
    affluence_list = np.random.choice(a, size=n, p=p)
    
    # Update the agents DataFrame with affluence values
    agents_df['affluence'] = affluence_list

    return agents_df


def calculate_amenity_influence(agents_df, amenities_df, amenity_weights, amenity_radius):
    '''
    Calculate the influence of nearby amenities on house prices based on radial distances.
    
    Inputs:
    agents_df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'price']
    amenities_df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'type']
    amenity_weights (dict): Dictionary mapping amenity types to their weight on house prices
    amenity_radius (float): Radius within which amenities influence house prices (in kilometers)
    
    Returns:
    amenity_influence (array): Array of influence values for each property in the DataFrame
    '''
    df = agents_df.copy()
    # Extract property and amenity coordinates
    property_coords = df[['latitude', 'longitude']].values
    amenity_coords = amenities_df[['latitude', 'longitude']].values
    amenity_types = amenities_df['Type'].values

    # Create spatial index for amenities
    amenity_tree = cKDTree(amenity_coords)

    # Initialize an array to hold amenity influence values
    influence = np.zeros(len(df))

    # Iterate over each property
    for i, (lat, lon) in enumerate(property_coords):
        # Find amenities within the radius
        nearby_idx = amenity_tree.query_ball_point([lat, lon], amenity_radius)
        if len(nearby_idx) > 0:
            total_weight = sum(amenity_weights.get(amenity_types[idx], 0) for idx in nearby_idx)
            # Sum of total amenity contributions
            influence[i] = total_weight 
       
    return influence


def update_house_prices(agents_df, weight, radius, amenities_df, amenity_weights, amenity_radius):
    '''
    Update all house values by incorporating neighborhood average price and amenity influence.
    
    Parameters:
    agents_df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'affluence', 'price'].
    weight (float): Weight for neighborhood average in calculating updated price.
    radius (float): Radius within which to calculate neighborhood average (in kilometers).
    amenities_df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'type'].
    amenity_weights (dict): Dictionary mapping amenity types to their weights.
    amenity_radius (float): Radius within which amenities influence house prices (in kilometers).
    
    Returns:
    df (DataFrame): DataFrame with updated house prices.
    '''
    
    df = agents_df.copy()
    # Extract coordinates into a numpy array for spatial indexing
    coords = df[['latitude', 'longitude']].values
    
    # Create a cKDTree for fast spatial queries
    tree = cKDTree(coords)
    
    # Precompute amenity influence for each house
    amenity_influence = calculate_amenity_influence(df, amenities_df, amenity_weights, amenity_radius)
    
    # Query all neighbors within radius for each house at once
    neighbors_idx_dict = {}
    for idx, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        neighbors_idx = tree.query_ball_point([lat, lon], radius)
        neighbors_idx_dict[idx] = neighbors_idx
    
    # Calculate neighborhood average prices using a loop to handle varying list lengths
    neighborhood_avg = []
    for idx, row in df.iterrows():
        # Get neighbors for the current index
        neighbors_idx = neighbors_idx_dict[idx]
        
        # Collect the prices of neighbors
        neighborhood_vals = df.iloc[neighbors_idx]['price'].values
        
        # Calculate neighborhood average price if any neighbors are within the radius
        if len(neighborhood_vals) > 0:
            neighborhood_avg.append(np.mean(neighborhood_vals))
        else:
            neighborhood_avg.append(row['price'])  # Use current price if no neighbors

    # Convert neighborhood averages into a numpy array
    neighborhood_avg = np.array(neighborhood_avg)
    
    # Vectorized update of the price
    updated_prices = df['affluence'] + weight * neighborhood_avg + amenity_influence
    
    # Assign updated prices back to the DataFrame
    df['price'] = updated_prices

    return df

def random_positions(df, point_idx):
    '''
    Given a DataFrame and an index of a specific point, return indices of 10 random points
    from the DataFrame excluding the provided point.
    
    Inputs:
    df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'District', 'affluence', 'price']
    point_idx (int): index of the specific point to exclude from selection
    
    Returns:
    new_point_indices (list of int): indices of 10 random points selected from the DataFrame, excluding `point_idx`
    '''
    # Generate random indices excluding the point_idx
    df_size = len(df)  # This avoids recalculating df size repeatedly
    all_indices = np.arange(df_size)  # Create an array of all indices
    available_indices = np.delete(all_indices, point_idx)  # Efficiently exclude the point_idx
    
    return np.random.choice(available_indices, size=10, replace=False)  # Randomly sample 10 indices

def perform_swaps(agent_df, point_idx, random_indices):
    '''
    Compute deltas for 10 random points compared to a specific point and swap with the point 
    that has the largest delta if it improves the metric.
    
    Inputs:
    agent_df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'District', 'affluence', 'price']
    point_idx (int): index of the specific point to compare with
    
    Returns:
    df (DataFrame): Updated DataFrame after performing the swap (if applicable)
    '''
    df = agent_df.copy()
    # Efficient extraction of values for the selected point
    affluence_x = df.at[point_idx, 'affluence']
    price_x = df.at[point_idx, 'price']

    # Efficient extraction of 10 random points
    affluences_y = df.loc[random_indices, 'affluence'].values
    prices_y = df.loc[random_indices, 'price'].values

    # Compute deltas using vectorized operations
    deltas = ((affluence_x - price_x) ** 2 + (affluences_y - prices_y) ** 2
              - (affluence_x - prices_y) ** 2 - (affluences_y - price_x) ** 2)

    # Find the index of the largest delta
    max_delta_idx = random_indices[np.argmax(deltas)]

    # Perform swap if the largest delta is positive
    if deltas[np.argmax(deltas)] > 0:
        df.loc[point_idx, 'affluence'], df.loc[max_delta_idx, 'affluence'] = (
            df.loc[max_delta_idx, 'affluence'], df.loc[point_idx, 'affluence']
        )
        return df, 1

    return df, 0

def new_iteration(df, weight, radius, amenities_df, amenity_weights, amenity_radius):
    '''
    Perform swaps on the affluence grid and calculate the new house values, including amenity influence.
    
    Inputs:
    df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'affluence', 'price']
    weight (float): Weight factor for updating house prices
    radius (float): Radius within which to calculate neighborhood average
    amenities_df (DataFrame): DataFrame with columns ['latitude', 'longitude', 'type']
    amenity_weights (dict): Dictionary mapping amenity types to their weights
    amenity_radius (float): Radius within which amenities influence house prices
    
    Returns:
    house_vals (array): Updated grid of house values
    affluence_grid (array): Updated grid of affluence values
    '''
    swap_tot = 0
    df = df.copy()
    # Generate random positions for all agents at once (excluding self)
    all_indices = df.index.values
    random_indices_all = np.array([np.random.choice(np.delete(all_indices, i), size=10, replace=False)
                                  for i in range(len(df))])

    # Perform all swaps in one go
    for idx1, random_indices in zip(df.index, random_indices_all):
        df, swap_num = perform_swaps(df, idx1, random_indices)
        swap_tot += swap_num

    # Update house prices after all swaps
    house_vals = update_house_prices(df, weight, radius, amenities_df, amenity_weights, amenity_radius)
    
    return house_vals, df['affluence'].values, swap_tot


def grid_initializer(num_agents, num_amenities, pop_dist, affluence_vals, affluence_dist, residence_df, amenities_df, mult = False):
    """
    Initializes the grid by placing agents and amenities within a specified set of districts. 

    This function distributes agents and amenities across different districts based on population 
    distribution and affluence values. The agents and amenities are placed in residential areas, and 
    each agent is assigned an initial affluence distribution. The function also calculates the appropriate 
    radius for agent placement based on the horizontal distance of the agent spread.

    Parameters:
    -----------
    num_agents : int
        The total number of agents to be placed in the grid.
    
    num_amenities : int
        The total number of amenities to be placed in the grid.

    pop_dist : pd.Series
        A series containing the population distribution across districts.

    affluence_vals : np.array
        An array containing different affluence values.

    affluence_dist : np.array
        An array representing the initial affluence distribution for agents.

    residence_df : pd.DataFrame
        A dataframe containing the residential areas data, including district and residential area geometries.

    amenities_df : pd.DataFrame
        A dataframe containing amenity data, with columns for geometry, Type, and District.
    mult: (int)
        Used to test if initial affluence distribution matters. Returns mult number of initial affluence grids. By default False.
    Returns:
    --------
    agents_df : GeoDataFrame
        A GeoDataFrame containing the coordinates and district of each placed agent, with an added 'affluence' column.
    
    amenities_df : DataFrame
        A DataFrame containing the coordinates and type of each placed amenity.
    
    radius : float
        The calculated radius, used for determining the spread of agents in the grid.
    
    """
    residence_df = residence_df.copy()
    amenities_df = amenities_df.copy()
    # Initialize agent and amenity lists
    agents = []
    amenities = []

    # AGENT PLACING:

    # Iterate over districts and place agents within residential areas in each district
    total_allocated_agents = 0
    districts = residence_df['District'].unique()  # List of unique districts from the residence data
    
    for district in districts:
        
        allocated_agents = 0  # Initialize the count of agents allocated in this district
        
        # Calculate the number of agents to allocate in each district based on population distribution
        agents_per_district = float(pop_dist.loc[district]) * num_agents
        
        # Get residential areas in the current district
        sub_residence = residence_df[residence_df['District'] == district]
        
        total_areas_in_district = len(sub_residence)  # Total number of residential areas in this district
        
        # Calculate the interval between agent placements
        skip_interval_agents = total_areas_in_district // agents_per_district
        
        # Place agents at intervals within residential areas in the district
        for i, (_, row) in enumerate(sub_residence.iterrows()):
            if i % skip_interval_agents == 0 and allocated_agents < agents_per_district and total_allocated_agents < num_agents:
                
                # Append the coordinates of the residential area to the agent list
                agents.append((row['longitude'], row['latitude'], 'agent', district))
                allocated_agents += 1  
                total_allocated_agents += 1   

    # AMENITY PLACING:

    # Initialize amenity distribution by district
    amenity_dist = {district: 0 for district in districts}
    
    # Calculate the amenity distribution for each district
    for district in districts:
        count_k = amenities_df['District'].value_counts().get(district, 0)
        amenity_dist[district] = count_k / len(amenities_df)
    
    # Initialize allocated amenity counter
    placed_amen = 0
    amenities = pd.DataFrame(columns=['longitude', 'latitude', 'Type'])

    # Distribute amenities across districts
    for n, district in enumerate(districts):
        
        allocated_amenities = 0
        amenities_per_district = np.round(amenity_dist[district] * num_amenities)
        
        sub_amenity = amenities_df[amenities_df['District'] == district]
        
        total_areas_in_district = len(sub_amenity)

        # Calculate relative amenity spread across different types
        relative_amenity_spread = []
        amenity_types = amenities_df['Type'].unique()
        
        for amenity_type in amenity_types:
            amenity_type_prop_per_district = np.round((sub_amenity['Type'].value_counts().get(amenity_type, 0)) / len(sub_amenity), 2)
            relative_amenity_spread.append(amenity_type_prop_per_district)
        
        unassigned_prop = np.round(1 - np.sum(relative_amenity_spread), 2)
        
        # Adjust distribution of amenities so that the sum of proportions equals 1
        while unassigned_prop != 0:
            min_idx = np.argmin(relative_amenity_spread)
            relative_amenity_spread[min_idx] = np.round(relative_amenity_spread[min_idx] + np.round(1 - np.sum(relative_amenity_spread), 2) / 2, 4)
            unassigned_prop = np.round(1 - np.sum(relative_amenity_spread), 2)
        
        relative_amenity_spread_array = np.array(relative_amenity_spread) * amenities_per_district
        rasa_avg = np.mean(relative_amenity_spread_array)
        
        # Round up or down based on the average to ensure correct number of amenities
        for i in range(len(relative_amenity_spread_array)):
            if relative_amenity_spread_array[i] >= rasa_avg:
                relative_amenity_spread_array[i] = np.floor(relative_amenity_spread_array[i])
            else:
                relative_amenity_spread_array[i] = np.ceil(relative_amenity_spread_array[i])
                
        relative_amenity_spread_array = relative_amenity_spread_array.astype(np.int32)
        sum_after_spread = np.sum(relative_amenity_spread_array)
        
        placed_amen += sum_after_spread
        
        # Adjust the last district if needed to match the total number of amenities
        if n == 4:
            current_dif = num_amenities - placed_amen
            max_idx = np.argmax(relative_amenity_spread_array)
            relative_amenity_spread_array[max_idx] = relative_amenity_spread_array[max_idx] + current_dif
            placed_amen += current_dif
        
        # Place the amenities based on the calculated spread
        for amenity_idx, amenity_type in enumerate(amenity_types):
            sub_amenity_type = sub_amenity[sub_amenity['Type'] == amenity_type]
            amenity_count = relative_amenity_spread_array[amenity_idx]
            
            if len(sub_amenity_type) == 0 or amenity_count == 0:
                continue
            
            selected_rows = sub_amenity_type.sample(n=min(amenity_count, len(sub_amenity_type)))
            
            # Add the selected amenity's location to the amenities DataFrame
            for _, row in selected_rows.iterrows():
                if allocated_amenities <= amenities_per_district:
                    centroid = row['geometry'].centroid
                    amenities = pd.concat(
                        [
                            amenities, 
                            pd.DataFrame({'longitude': [centroid.x], 'latitude': [centroid.y], 'Type': [amenity_type]})
                        ], 
                        ignore_index=True
                    )
                    allocated_amenities += 1

    # Assign districts to agents based on their coordinates
    agents_df = pd.DataFrame(agents, columns=["longitude", "latitude", "type", "District"]).drop(columns=["type"])

    # Create a geometry column for the agents
    agents_df['geometry'] = agents_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    
     # Calculate the radius based on agent and amenity spread
    agent_coords = agents_df[['longitude', 'latitude']].values
    max_distance = max(distance.cdist(agent_coords, agent_coords, metric="euclidean").flatten())
    radius_mod = max((num_agents+ num_amenities)/100, 5)
    radius = np.round(max_distance / radius_mod, 5)
    
    # Assumption that amenity radius is smaller than neighborhood radius. For 100 agent model this translates to a roughly 5 km agent radius and 1 km amenity radius which
    # works very well with our model.
    radius_amenity = radius / 5
    
    if mult == False:
        # Add initial affluence distribution to agents
        agents_df = initial_affluence_dist(num_agents, affluence_vals, affluence_dist, agents_df)
        
        return agents_df, amenities, radius, radius_amenity
    
    else:
        affluences = []
        for i in range(mult):
            # Add initial affluence distribution to agents
            agent_affluence = initial_affluence_dist(num_agents, affluence_vals, affluence_dist, agents_df)
            affluences.append(agent_affluence)

        return affluences, amenities, radius, radius_amenity

def available_residences(agent_df, district_list, residences, growth_rate):
    '''
    Computes available residences after inizialization.
    
    takes:
    
    agent_df: dataframe with model agents, columns latitude, longitude, District, affluence, price
    
    district_list: ordered list of districts
    
    residences: dataframe with residence locations.
    growth_rate: dataframe with percentage growth rate for each district
    
    returns:
    
    available_residences: empty locations where agents have not yet been placed
    
    yearly_agents_added: array of agents to be added inside iteration loop
    
    '''
    agent_df = agent_df.copy()
    residences = residences.copy()
    
    # Calculate centroids for all residence df
    agents_with_districts = agent_df.reset_index(drop = True)

    # Convert latitude and longitude into tuples for comparison
    residence_tuples = set(residences[['latitude', 'longitude']].apply(tuple, axis=1))
    agents_tuples = set(agents_with_districts[['latitude', 'longitude']].apply(tuple, axis=1))

    # Identify unique points in geographic_residence not in geographic_df

    unique_tuples = list(residence_tuples - agents_tuples)

    # Filter geographic_residence based on unique tuples
    available_residences = residences[
        residences[['latitude', 'longitude']].apply(tuple, axis=1).isin(unique_tuples)
    ]

    # Compute number of agents needed at each year addition
    yearly_agents_added = []

    for district in district_list:

        # Extract growth rate for borough k and divide by 4 to get yearly rate
        rate = (1 + float(growth_rate.loc[district])) ** (1/4) -1
        
        # Calculate number of agents to be added to each district
        num_agents_in_district = agent_df['District'].value_counts().get(district, 0)
        num_agents_added = int(np.ceil(num_agents_in_district * rate))
        yearly_agents_added.append(num_agents_added)
        
    return available_residences, yearly_agents_added
   

def add_agents(agents_df, radius, agents_needed, residences, a, p, year, price_data, type='median'):
    '''
    Adds agents to a dataset based on growth rates and price data for each borough.

    Parameters:
    agents_df (DataFrame): Existing agents' DataFrame with columns 'latitude', 'longitude', 'District', and 'price'.
    radius (float): Radius (in kilometers) to consider for calculating neighborhood price averages.
    agents_needed (list): List specifying the number of agents needed for each district.
    residences (DataFrame): DataFrame with residential areas, including 'latitude', 'longitude', and 'District'.
    a (array-like): Array of affluence levels.
    p (array-like): Probabilities corresponding to the affluence levels.
    year (int): Year of data to be considered for price calculations.
    price_data (DataFrame): DataFrame containing yearly price information indexed by district.
    type (str, optional): Type of price assignment - 'mean', 'median', or 'actual' (default is 'median').

    Returns:
    DataFrame: Updated agents_df with newly added agents and their calculated 'price' and 'affluence'.
    '''
    agents_df = agents_df.copy()
    # Extract price data for the specified year
    year_prices = price_data[price_data.index == year]
    
    # Validate the 'type' parameter
    if type not in ['mean', 'median', 'actual']:
        raise ValueError("Invalid type. Use 'mean', 'median', or 'actual'.")
    
    # Determine the price value based on the selected 'type'
    if type == 'mean':
        val = np.mean(year_prices)
    elif type == 'median':
        val = np.median(year_prices)
    else:  # 'actual'
        year_prices = year_prices.to_dict()
        val = agents_df['District'].map(year_prices)
    
    # Extract agent coordinates into a NumPy array for spatial indexing
    coords = agents_df[['latitude', 'longitude']].values
    
    # Create a cKDTree for fast spatial queries
    tree = cKDTree(coords)
    
    # Get the list of unique districts in the residence data
    districts = residences['District'].unique()
    
    for n, district in enumerate(districts):
        # Initialize the count of allocated agents for the district
        allocated_agents = 0
        
        # Number of agents needed for the current district
        agents_per_district = agents_needed[n]
        
        # Filter residential areas for the current district
        sub_residence = residences[residences['District'] == district]
        
        # Total number of residential areas in the district
        total_areas_in_district = len(sub_residence)
        
        # Calculate the interval to skip between agent placements
        skip_interval_agents = total_areas_in_district // agents_per_district
        
        # Iterate over residential areas in the district
        for i, (_, row) in enumerate(sub_residence.iterrows()):
            # Place an agent at every 'skip_interval_agents'-th area
            # Ensure the total number of agents does not exceed the desired count
            if i % skip_interval_agents == 0 and allocated_agents < agents_per_district:
                # Randomly select an affluence level based on the distribution
                affluence = np.random.choice(a, size=1, p=p)
                
                # Get the latitude and longitude of the residence
                lat, lon = row['latitude'], row['longitude']
                
                # Query the cKDTree for neighbors within the specified radius
                neighbors_idx = tree.query_ball_point([lat, lon], radius)
                
                # Get the prices of neighbors within the radius
                neighborhood_vals = agents_df.iloc[neighbors_idx]['price'].values
                
                # Calculate the neighborhood average price if neighbors are found
                if len(neighborhood_vals) > 0:
                    neighborhood_avg = np.mean(neighborhood_vals)
                else:
                    # Use the fallback price value if no neighbors are within the radius
                    neighborhood_avg = val
                
                # Create a new agent with the calculated attributes
                agent = pd.DataFrame({
                    'longitude': [lon],
                    'latitude': [lat],
                    'District': [district],
                    'price': [neighborhood_avg],
                    'affluence': affluence
                })
                
                # Append the new agent to the agents DataFrame
                agents_df = pd.concat([agents_df, agent], ignore_index=True)
                allocated_agents += 1
    
    # Return the updated agents DataFrame
    return agents_df

def calculate_mse(array1, array2):
    """
    Calculate the Mean Squared Error (MSE) between two arrays.
    
    Parameters:
    array1 (np.ndarray): First array of numerical values.
    array2 (np.ndarray): Second array of numerical values (of the same length as array1).
    
    Returns:
    float: The calculated Mean Squared Error (MSE).
    """
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Both arrays must have the same length")
    
    # Calculate the squared differences between corresponding elements
    squared_diff = (array1 - array2) ** 2
    
    # Return the mean of the squared differences
    mse = np.mean(squared_diff)
    return mse


def calculate_ape(array1, array2):
    """
    Calculate the Average Percentage Error (APE) between two arrays.
    
    Parameters:
    array1 (np.ndarray): First array of numerical values.
    array2 (np.ndarray): Second array of numerical values (of the same length as array1).
    
    Returns:
    float: The calculated Average Percentage Error (APE).
    """
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Both arrays must have the same length")
    
    # Calculate the absolute percentage errors between corresponding elements
    ape = np.abs((array1 - array2) / array2) * 100
    
    return ape







