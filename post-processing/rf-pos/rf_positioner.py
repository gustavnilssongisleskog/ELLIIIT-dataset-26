import rf_local_functions as rflf 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    requests = {
        #"EXP003": [1, 100, 200, 300, 400],
        "EXP006": [1],
        #"EXP011": [51, 55, 60]
    }
    ds = rflf.extract_phase(requests)
    #rflf.estimate_positions_2d_from_phase(ds, frequency_hz=920e6, height_offset=2.4-0.75)
    #room is 8.56x4x2.4
    # force it to only consider ranges from 2.4-0.75 (the rover height to the roof)
    # to 8.56 (the room length) to avoid errors with the 2d projection
    fig,ax = rflf.plot_position_candidates(ds, 
                                           observation_index=0, 
                                           frequency_hz=920e6, 
                                           height_offset=2.4-0.75, 
                                           min_distance=2.4-0.75, 
                                           max_distance=8.56
                                           )
    plt.show()

# Strunta i EXP12 och EXP5 för de är konstiga
# # usrp settings
# frequency: 920e6
# channel: 0
# gain: 80
# rate: 250e3
# duration : 36000                      # time span over which phases are kept constant
# #duration : 10                       # time span over which phases are kept constant (run_adaptive_single_tone)