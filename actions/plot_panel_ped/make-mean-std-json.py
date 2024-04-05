import json
import numpy as np
import argparse
import h5py
_default_input_file=None
_default_file_prefix=None



def main(input_file=_default_input_file, \
         file_prefix=_default_file_prefix, \
         **kwargs):
    if input_file==None:
        print('Provide HDF5 packet input file. Exiting.')
        return
    if file_prefix==None:
        print('Provide output filename string. Exiting.')
        return
    d={}
    files = []
    files.append(input_file)    
    for input_file in files:
        f = h5py.File(input_file)
        packets=f['packets']
        packets=packets[packets['valid_parity']==1]
        packets=packets[packets['packet_type']==0]    
        io_group = set(packets['io_group'])
        for iog in io_group:
            _packets = packets[packets['io_group']==iog]
            io_channel = set(_packets['io_channel'])
            for ioc in io_channel:
                __packets = _packets[_packets['io_channel']==ioc]
                chip_id = set(__packets['chip_id'])
                for cid in chip_id:
                    ___packets = __packets[__packets['chip_id']==cid]
                    channel = set(___packets['channel_id'])
                    for chan in channel:
                        data=___packets[___packets['channel_id']==chan]
                        key='{}-{}-{}-{}'.format(iog, ioc, cid, chan)
                        if not key in d: d[key]={'n':0,'sum':0,'sum2':0}
                        d[key]['n']+=len(data)
                        d[key]['sum']+=np.sum(data['dataword'].astype(float))
                        d[key]['sum2']+=np.sum(np.square(data['dataword'].astype(float)))

    datawords, stds = [[] for i in range(2)]
    for key in d.keys():
        datawords.append(d[key]['sum']/d[key]['n'])
        d[key]['mean']=d[key]['sum']/d[key]['n']
        d[key]['std']=np.sqrt(-1*(d[key]['mean'])**2+d[key]['sum2']/d[key]['n'] )
        stds.append(d[key]['std'])

    output={}
    for key in d.keys():
        ids=key.split('-')
        newkey='{}-{}-{}'.format(ids[0], ids[1], ids[2])
        if newkey not in output.keys():
            output[newkey]=[(-1,-1) for i in range(64)]
        output[newkey][int(ids[-1])]=(d[key]['mean'], d[key]['std'])
        
    with open(file_prefix+'.json','w') as fout:
        json.dump(output, fout, indent=4)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=_default_input_file, \
                        type=str, help='''HDF5 packet file''')
    parser.add_argument('--file_prefix', default=_default_file_prefix, \
                        type=str, help='''String prepended to file''')
    args=parser.parse_args()
    main(**vars(args))

