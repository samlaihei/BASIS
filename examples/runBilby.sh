python runBilby.py --bilby_label="test_xsring" \
                --uvfits="../data/uvfits/test_xsring.uvfits" \
                --models="xsring" \
                --imgdim=128 \
                --fov=225 \
                --ncpu=64 \
                --static_noise_floor=0.0001 \
                --noise_frac=0.05 \
                --noise_factor=1.0 \
                --bilby_sampler="dynesty" \
                --bilby_npoints=250 \
                --random_seed=42 \
                --data_terms="ci" \
                --data_weights="100" 


# M87 EXAMPLE

# python runBilby.py --bilby_label="SR1_M87_2017_101_hilo_hops_netcal_StokesI_xsrggg" \
#                 --uvfits="../data/uvfits/SR1_M87_2017_101_hilo_hops_netcal_StokesI.uvfits" \
#                 --models="xsringauss,gauss,gauss" \
#                 --imgdim=128 \
#                 --fov=225 \
#                 --ncpu=64 \
#                 --static_noise_floor=0.0001 \
#                 --noise_frac=0.05 \
#                 --noise_factor=1.0 \
#                 --bilby_sampler="dynesty" \
#                 --bilby_npoints=250 \
#                 --random_seed=42 \
#                 --data_terms="ci" \
#                 --data_weights="100" 
