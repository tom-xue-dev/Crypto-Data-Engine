from read_large_files import load_filtered_data_as_list,select_assets

start = "2020-5-1"
end = "2020-6-1"
asset = select_assets(spot=True,n=3)
data = load_filtered_data_as_list(start, end, asset, level="1d")


print(data)
