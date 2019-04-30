import click
from . import cli
import clodius.chromosomes as cch
import clodius.multivec as cmv
import gzip
import h5py
import math
import negspy.coordinates as nc
import numpy as np
import os
import os.path as op
import scipy.misc as sm
import tempfile
import json
import sys

import ast

def epilogos_bedline_to_vector(bedlines, row_infos=None):
    '''
    Convert a line from an epilogos bedfile to vector format.

    Parameters
    -----------
    bedline: [string,....]
        A line from a bedfile broken up into its constituent parts
        (e.g. ["chr1", "1000", "2000", "[1,2,34,5]"])

    Returns
    -------
    An array containing the values associated with that line
    '''
    bedline = bedlines[0]
    parts = bedline.decode('utf8').strip().split('\t')
    # extract the state values e.g. [...,[0,14],[0.56,15]]
    array_str = parts[3].split(':')[-1]

    # make sure they're ordered by their index
    array_val = sorted(ast.literal_eval(array_str), key=lambda x: x[1])
    states = [v[0] for v in array_val]

    chrom=parts[0]
    start=int(parts[1])
    end=int(parts[2])

    return (chrom, start, end, states)

def states_bedline_to_vector(bedlines, states_dic):
    '''
    Convert a line from a bedfile containing states in categorical data to vector format.

    Parameters
    ----------

    bedline: [string,...]
        A line form a bedfile broken up into its contituent parts
        (e.g. ["chr1", "1000", "2000", "state"]))


    states_dic: {'key':val,...}
        A dictionary containing the states in the file with a corresponding value
        (e.g. {'state1_name': 1, 'state2_name': 2,...})

    Returns
    -------

    Four variables containing the values associated with that line: chrom, start, end, states_vector
    (e.g. chrom = "chr1", start = 1000, end = 2000, states_vector = [1,0,0,0])
    '''
    # we support passing in multiple bed files for multivec creation from
    # # other file types, but this one only supports a single file so just
    # # assume that a single file is passed in
    bedline = bedlines[0]

    parts = bedline.decode('utf8').strip().split('\t')
    chrom=parts[0]
    start=int(parts[1])
    end=int(parts[2])
    state= states_dic[parts[3].encode('utf8')]

    states_vector = [ 1 if index == state else 0 for index in range(len(states_dic))]

    return (chrom, start, end, states_vector)

def categorical_bedline_to_vector(bedlines, row_infos=None):
    '''
    Convert a line from a categorical bedfile to vector format.

    One example of categorical data includes chromatin state marks for 
    visualizing a 2D matrix of samples and bins.

    In this example, marks are an ordered list of chromatin state assignments,
    one for each sample. Each state within that ordered list corresponds to an 
    epigenomic sample label (e.g., E123, E105, etc.).

    A state is a discrete integer label, from '1' to '15' for a 15-state
    chromatin model, '1' to '18' for an 18-state model, and so on.

    These integer labels or keys map back to indices for a color-map, and 
    labeling of categories should start at '1'.

    Parameters
    -----------
    bedline: [string,....]
        A line from a bedfile broken up into its constituent parts
        (e.g. ["chr1", "1000", "2000", "[1,2,3,4,5]"])

    Returns
    -------
    An array containing the values associated with that line
    '''
    bedline = bedlines[0]
    parts = bedline.decode('utf8').strip().split('\t')
    # extract the state values e.g. [3,12,1,...,15]
    array_str = parts[3]

    # categories are a literal evaluation of the array string
    categories = ast.literal_eval(array_str)

    chrom = parts[0]
    start = int(parts[1])
    end = int(parts[2])

    return (chrom, start, end, categories)

def mode(ndarray, axis=0):
    '''
    A faster mode than scipy.stats.mode
    https://stackoverflow.com/a/35674754/19410
    '''
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on an empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)

    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1

    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                              np.diff(sort, axis=axis) == 0,
                              np.zeros(shape=shape, dtype='bool')],
                             axis=axis).transpose(transpose).ravel()
    
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0

    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)

    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[tuple(slices)] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]

@cli.group()
def convert():
    '''
    Aggregate a data file so that it stores the data at multiple
    resolutions.
    '''
    pass

def _bedgraph_to_multivec(
        filepaths,
        output_file,
        assembly,
        chrom_col,
        from_pos_col,
        to_pos_col,
        value_col,
        has_header,
        chunk_size,
        nan_value,
        chromsizes_filename,
        starting_resolution,
        num_rows,
        format,
        row_infos_filename,
        category_infos_filename,
        background_freqs_filename,
        tile_size,
        method,
        bin_span,
        bin_fill
):
    print('chrom_col:', chrom_col)

    with tempfile.TemporaryDirectory() as td:
        print('temporary dir:', td)

        temp_file = op.join(td, 'temp.mv5')
        f_out = h5py.File(temp_file, 'w')

        (chrom_info, chrom_names, chrom_sizes) = cch.load_chromsizes(chromsizes_filename, assembly)

        if row_infos_filename is not None:
            row_infos = []
            with open(row_infos_filename, 'r') as fr:
                row_infos = [l.strip().encode('utf8') for l in fr]
                #row_infos_obj = json.load(fr)
                #row_infos = row_infos_obj['row_infos']

        else:
            row_infos = None

        if category_infos_filename is not None:
            with open(category_infos_filename, 'r') as fr:
                try:
                    category_infos = json.load(fr)
                    category_count = str(len(category_infos))
                except json.decoder.JSONDecodeError:
                    sys.stderr.write("Error: Could not decode category-infos-filename (is it JSON-formatted?)\n")
                    sys.exit(-1)
        else:
            category_infos = None
            category_count = None

        if background_freqs_filename is not None:
            with open(background_freqs_filename, 'r') as fr:
                background_freqs = json.load(fr)
        else:
            background_freqs= None

        for chrom in chrom_info.chrom_order:
            f_out.create_dataset(chrom, 
                                 (math.ceil(chrom_info.chrom_lengths[chrom] / starting_resolution), num_rows * len(filepaths)),
                                 fillvalue=bin_fill,
                                 compression='gzip')

        def bedline_to_chrom_start_end_vector(bedlines, row_infos=None):
            chrom_set = set()
            start_set = set()
            end_set = set()
            all_vector = []
 
            for bedline in bedlines:
                parts = bedline.strip().split()
                chrom = parts[chrom_col-1]
                start = int(parts[from_pos_col-1])
                end = int(parts[to_pos_col-1])
                vector = [float(f) if not f == 'NA' else np.nan 
                        for f in parts[value_col-1:value_col-1+num_rows]]                
                chrom_set.add(chrom)
                start_set.add(start)
                end_set.add(end)

                if len(chrom_set) > 1:
                    raise ValueError("Chromosomes don't match in these lines:", bedlines)
                if len(start_set) > 1:
                    raise ValueError("Start positions don't match in these lines:", bedlines)
                if len(end_set) > 1:
                    raise ValueError("End positions don't match in these lines:", bedlines)
                all_vector += vector

            return (list(chrom_set)[0],
                    list(start_set)[0],
                    list(end_set)[0],
                    all_vector)

        if format == 'epilogos':
            cmv.bedfile_to_multivec(filepaths, f_out, epilogos_bedline_to_vector,
                                    starting_resolution, has_header, chunk_size)
        elif format == 'states':
            assert(row_infos != None), "A row_infos file must be provided for --format = 'states' "
            states_dic = {row_infos[x]:x for x in range(len(row_infos))}

            cmv.bedfile_to_multivec(filepaths, f_out, states_bedline_to_vector,
                                    starting_resolution, has_header, chunk_size, 
                                    states_dic)
        elif format == 'categorical':
            assert(row_infos != None), "A row_infos file must be provided for --format = 'categorical' "
            assert(category_infos != None), "A category_infos file must be provided for --format = 'categorical' "
            cmv.bedfile_to_multivec(filepaths, f_out, categorical_bedline_to_vector,
                                    starting_resolution, has_header, chunk_size)
        else:
            cmv.bedfile_to_multivec(filepaths, f_out, bedline_to_chrom_start_end_vector,
                                    starting_resolution, has_header, chunk_size)

        f_out.close()
        tf = temp_file
        f_in = h5py.File(tf, 'r')

        if output_file is None:
            output_file = op.splitext(filepaths[0])[0] + '.multires.mv5'
        print('output_file:', output_file)

        # Override the output file if it existts
        if op.exists(output_file):
            os.remove(output_file)

        if method =='logsumexp':
            def agg(x):
                # newshape = (x.shape[2], -1, 2)
                # b = x.T.reshape((-1,))

                a = x.T.reshape((x.shape[1],-1,2))

                # this is going to be an odd way to get rid of nan
                # values
                orig_shape = a.shape
                na = a.reshape((-1,))

                SMALL_NUM = -1e8
                NAN_THRESHOLD_NUM = SMALL_NUM / 100

                if np.nanmin(na) < NAN_THRESHOLD_NUM:
                    raise ValueError("Error removing nan's when running logsumexp aggregation")

                na[np.isnan(na)] = SMALL_NUM;
                na = na.reshape(orig_shape)
                res = sm.logsumexp(a, axis=2).T

                nres = res.reshape((-1,))
                # print("nres:", np.nansum(nres < NAN_THRESHOLD_NUM))
                nres[nres < NAN_THRESHOLD_NUM] = np.nan
                res = nres.reshape(res.shape)

                # print("res:", np.nansum(res.reshape((-1,))))

                return res

        elif method == 'sum':
            agg = lambda x: x.T.reshape((x.shape[1], -1, 2)).sum(axis=2).T

        elif method == 'mode':
            def agg(x):
                rows = x.shape[0]
                cols = x.shape[1]
                left_span = math.floor(bin_span/2)
                right_span = math.ceil(bin_span/2)
                assert(left_span + right_span == bin_span)
                res = np.zeros((int(x.shape[0]/2), x.shape[1]))
                ri = 0
                # step every second row, take the mode (except at edges)
                for i in range(0, rows, 2):
                    l = i - left_span
                    r = i + right_span
                    if l < 0: l = 0
                    if r > (rows - 1): r = (rows - 1)
                    y = x[l:r].T
                    if y.shape[1] % 2 == 0:
                        res[ri,] = y.T[y.shape[1]-1,]
                    else:
                        res[ri,] = mode(y, axis=1)[0]
                    assert(res[ri,].shape[0] == cols)
                    ri += 1
                return res

        elif method == 'background-freqs':
            def agg(x):
                rows = x.shape[0]
                cols = x.shape[1]
                left_span = math.floor(bin_span/2)
                right_span = math.ceil(bin_span/2)
                assert(left_span + right_span == bin_span)
                res = np.zeros((int(x.shape[0]/2), x.shape[1]))
                ri = 0
                # step every second row, choosing the category with the lowest background frequency 
                # within the sample-group, as this is least likely to be observed within the sample and 
                # is therefore the "most interesting" to show at larger scales
                for i in range(0, rows, 2):
                    l = i - left_span
                    r = i + right_span
                    # handle chunk edges with smaller window
                    if l < 0: l = 0
                    if r > (rows - 1): r = (rows - 1)
                    y = x[l:r].T

                    z = np.zeros(cols).astype(int)
                    for e_idx, e in enumerate(y):
                        try:
                            row_id = row_infos[e_idx].decode().split('|')[0].rstrip() # e.g., sample name ("E017 | IMR90 fetal lung fibroblasts", etc.)
                            #row_id = row_infos[e_idx]['name']
                            freqs = background_freqs[assembly][category_count][row_id]
                        except KeyError:
                            sys.stderr.write('KeyError discovered in background_freqs\n')
                            sys.stderr.write('{}\n'.format(row_infos[e_idx]))
                            sys.stderr.write('{}\n'.format(row_id))
                            sys.stderr.write('{}\n'.format(assembly))
                            sys.stderr.write('{}\n'.format(category_count))
                            sys.stderr.write('{}\n'.format(background_freqs[assembly][category_count]))
                            sys.exit(-1)
                        min_freq = 1
                        min_freq_idx = -1
                        for v_idx, v in enumerate(e):
                            obs_freq = float(freqs[str(int(v))])
                            if obs_freq < min_freq:
                                min_freq = obs_freq
                                min_freq_idx = v_idx
                        z[e_idx] = e[min_freq_idx]

                    res[ri,] = z
                    assert(res[ri,].shape[0] == cols)
                    ri += 1
                return res

        else:
            raise ValueError("Specified aggregation method is unknown:", method)

        cmv.create_multivec_multires(f_in,
                                     chromsizes = zip(chrom_names, chrom_sizes),
                                     agg=agg,
                                     starting_resolution=starting_resolution,
                                     tile_size=tile_size,
                                     output_file=output_file,
                                     row_infos=row_infos,
                                     category_infos=category_infos)

@convert.command()
@click.argument(
    'filepaths',
    metavar='FILEPATHS',
    nargs=-1
)
@click.option(
    '--output-file',
    '-o',
    default=None,
    help="The default output file name to use. If this isn't"
         "specified, clodius will replace the current extension"
         "with .hitile"
)
@click.option(
    '--assembly',
    '-a',
    help='The genome assembly that this file was created against',
    type=click.Choice(nc.available_chromsizes()),
    default='hg19'
)
@click.option(
    '--chromosome-col',
    help="The column number (1-based) which contains the chromosome "
         "name",
    default=1,
    type=int
)
@click.option(
    '--from-pos-col',
    help="The column number (1-based) which contains the starting "
         "position",
    default=2,
    type=int
)
@click.option(
    '--to-pos-col',
    help="The column number (1-based) which contains the ending"
         "position",
    default=3,
    type=int
)
@click.option(
    '--value-col',
    help="The column number (1-based) which contains the actual value"
         "position",
    default=4,
    type=int
)
@click.option(
    '--has-header/--no-header',
    help="Does this file have a header that we should ignore",
    default=False
)
@click.option(
    '--chunk-size',
    help="The size of the chunks to read in at once",
    default=1e5
)
@click.option(
    '--nan-value',
    help='The string to use as a NaN value',
    type=str,
    default=None
)
@click.option(
    '--chromsizes-filename',
    help="A file containing chromosome sizes and order",
    default=None
)
@click.option(
    '--starting-resolution',
    help="The base resolution of the data. Used to determine how much space to allocate"
         " in the multivec file",
    default=1
)
@click.option(
    '--num-rows',
    help="The number of rows at each position in the multivec format",
    default=1
)
@click.option(
    '--format',
    type=click.Choice(['default', 'epilogos', 'states', 'categorical']),
    help= "'default':chr start end state1_value state2_value, etc; "
    "'epilogos': chr start end [[state1_value, state1_num],[state2_value, state2_num],[etc]]; "
    "'states': chr start end state_name; "
    "'categorical': chr start end [category1_num, category2_num, ...]",
    default='default'
)
@click.option(
    '--row-infos-filename',
    help="A file containing the names of the rows in the multivec file",
    default=None
)
@click.option(
    '--category-infos-filename',
    help="A file containing the names of the categories in the multivec file",
    default=None
)
@click.option(
    '--background-freqs-filename',
    help="If using \"--method background-freqs\", this file contains zero-order background "
    "frequencies of categories on a per-row (sample) basis",
    default=None
)
@click.option(
    '--tile-size',
    '-t',
    default=256,
    help="The number of data points in each tile."
         "Used to determine the number of zoom levels"
         "to create."
)
@click.option(
    '--method',
    help='The method used to aggregate values (e.g. sum, average...)',
    type=click.Choice(['sum', 'logsumexp', 'mode', 'background-freqs']),
    default='sum'
)
@click.option(
    '--bin-span',
    help='If using \"--method mode\", or \"--method background-freqs\", this value specifies the '
    'number of bins over which a mode or least-frequently-observed value is derived',
    default=5,
    type=int
)
@click.option(
    '--bin-fill',
    help='This value fills in rightmost edges of bins (default = NaN)',
    default=np.nan
)

def bedfile_to_multivec(filepaths, output_file, assembly, chromosome_col,
                        from_pos_col, to_pos_col, value_col, has_header,
                        chunk_size, nan_value,
                        chromsizes_filename,
                        starting_resolution, num_rows,
                        format, row_infos_filename, category_infos_filename, background_freqs_filename,
                        tile_size, method, bin_span, bin_fill):
    _bedgraph_to_multivec(filepaths, output_file, assembly, chromosome_col,
                          from_pos_col, to_pos_col, value_col, has_header,
                          chunk_size, nan_value,
                          chromsizes_filename, starting_resolution, num_rows,
                          format, row_infos_filename, category_infos_filename, background_freqs_filename,
                          tile_size, method, bin_span, bin_fill)
