import rf_local_functions as rflf 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    requests = {
        "EXP003": [1, 100, 200, 300, 400],
        "EXP005": [1, 2, 3],
        "EXP012": [51, 55, 60]
    }
    ds = rflf.extract_phase(requests)
    #rflf.estimate_positions_2d_from_phase(ds, frequency_hz=920e6, height_offset=2.4-0.75)
    #room is 8.56x4x2.4
    fig,ax = rflf.plot_position_candidates(ds, observation_index=0, frequency_hz=920e6, height_offset=2.4-0.75, max_distance=1)
    plt.show()


# # usrp settings
# frequency: 920e6
# channel: 0
# gain: 80
# rate: 250e3
# duration : 36000                      # time span over which phases are kept constant
# #duration : 10                       # time span over which phases are kept constant (run_adaptive_single_tone)