"""
Downloads and saves files from physionet.org to a local folder using a
the wfdb package. Also includes functions to ease reading ECG files including
plotting signals and their annotations, indexing instances of the wfdb Record
class, and converting from sample points to seconds.

Example:

    >>> import matplotlib.pyplot as plot
    >>> from retrieve_physio_files import get_record, plot_record
    >>> record_list = get_record_list('mitdb')
    >>> record = get_record(record_list[0], 'mitdb')
    >>> record = record.time_slice(1500, step=20)
    >>> plot_record(record)
    >>> plt.show()

"""

import os
import cPickle
import re
import csv
from StringIO import StringIO
import copy
import requests
import numpy as np
import matplotlib.pyplot as plt
import wfdb

#%% Change this line to change save location. %%#
SAVE_PATH = '/Volumes/PhysioNet' # macOS
# SAVE_PATH = '/run/media/david/PhysioNet' # manjaro
SAVE_PATH += '/RecordFiles'

# Add database name and annotation extension if missing.
ANNOTATION_EXTS = {'mghdb':'ari', 'mitdb':'atr', 'fantasia':'ecg',
                   'wrist':'atr', 'edb':'atr', 'cudb':'atr', 'afdb':'atr',
                   'svdb':'atr', 'sddb':'atr'}

# Print get_ann_symbol_table() for more information on annotation symbols.
IGNORED_ANNOTATIONS = ('"', '=', 'p', 't', '+', 'u', '?', '@', '~', '*', 'D')


def get_record_list(database):
    """ Returns a list of all file names for the database. """

    base_url = 'https://physionet.org/physiobank/database/'
    record_url = base_url + database + '/RECORDS'
    record_file = requests.get(record_url).content

    return re.findall(r'[\S]+', record_file)


class Record(wfdb.Record):
    """ Modifies wfdb.Record class to include annotations and adds __getitem__
    and time_slice methods. """

    def __init__(self, record, ann):
        super(Record, self).__init__(
            p_signal=record.p_signal,
            d_signal=record.d_signal,
            e_p_signal=record.e_p_signal,
            e_d_signal=record.e_d_signal,
            record_name=record.record_name,
            file_name=record.file_name,
            n_sig=record.n_sig,
            fs=record.fs,
            counter_freq=record.counter_freq,
            base_counter=record.base_counter,
            sig_len=record.sig_len,
            base_time=record.base_time,
            base_date=record.base_date,
            fmt=record.fmt,
            samps_per_frame=record.samps_per_frame,
            skew=record.skew,
            byte_offset=record.byte_offset,
            adc_gain=record.adc_gain,
            baseline=record.baseline,
            units=record.units,
            adc_res=record.adc_res,
            adc_zero=record.adc_zero,
            init_value=record.init_value,
            checksum=record.checksum,
            block_size=record.block_size,
            sig_name=record.sig_name,
            comments=record.comments
        )

        self.ann = ann

    def __getitem__(self, key):
        errmsg = 'Record __getitem__ only works with slices.'
        assert isinstance(key, slice), errmsg

        record_copy = copy.deepcopy(self)

        record_copy = self._get_ann_items(record_copy, key)
        record_copy = self._get_signal_items(record_copy, key)

        return record_copy

    @staticmethod
    def _get_ann_items(record, key):
        start = key.start
        stop = key.stop
        samples = record.ann.sample

        indeces = np.logical_and(samples >= start, samples <= stop)

        record.ann.sample = samples[indeces] - start
        record.ann.symbol = record.ann.symbol[indeces]

        return record

    @staticmethod
    def _get_signal_items(record, key):
        record_signals = (
            'p_signals',
            'd_signals',
            'e_p_signals',
            'e_d_signals'
        )

        for signal_name in record_signals:
            signal = getattr(record, signal_name)

            if signal is not None:
                indexed_signal = signal[:, key]
                setattr(record, signal_name, indexed_signal)

        return record

    def wrsamp(self, write_dir=''):
        return super(Record, self).wrsamp(write_dir=write_dir)

    def arrange_fields(self, channels, expanded=False):
        return super(Record, self).arrange_fields(channels, expanded=expanded)

    def time_slice(self, start, step=100):
        """ Slices instance of Record using time index (in seconds).

        Parameters:
            start (positive integer): the time (in seconds) to start the slice.

            step (positive integer): the length (in seconds) of the slice."""

        start_idx = start * self.fs
        stop_idx = start_idx + (step * self.fs)

        return self[start_idx:stop_idx]

def get_record(file_name, database, overwrite=False, save=True):
    """ Returns a class instance of wfdb class Record for file_name in
    database. The Record is loaded if it exists locally. Otherwise the Record is
    downloaded from the database then saved locally in the directory
    'path-to-file/database'.

    Parameters:
        file_name (str): The name of the record file.

        database (str): The name of the database containing the record.

        overwrite (optional bool): If True get_record will re-download the file
        from the database and write over the old value (if there is already a
        local copy otherwise, parameter is ignored). False by default.

        save (optional bool): If true the record will be saved to the folder
        named after the database in the directory containing this file. If the
        folder does not exist it will be created. True by default. """

    def get_annotations():
        """ Gets the record's annotations. """

        ann = wfdb.rdann(file_name, annotation_ext, pb_dir=database)
        signal_end = len(record.p_signal[0])
        indeces_within_signal = ann.sample < signal_end # Some annotations
        ann.sample = ann.sample[indeces_within_signal]  # go past signal.
        ann.symbol = np.array(ann.symbol)
        ann.symbol = ann.symbol[indeces_within_signal]

        return ann

    _check_arguments_are_strings(file_name, database)
    _if_missing_make_directory_for(database)

    if _is_file_in_directory(file_name, database) and not overwrite:
        return _load(file_name, database)

    try:
        annotation_ext = ANNOTATION_EXTS[database]

    except KeyError:
        err_msg = (
            'No annotation extension saved for %s. Add extension to'
            'ANNOTATION_EXTS dict.' % database
        )
        raise KeyError, err_msg

    try:
        print 'Downloading file %s...' % file_name
        sampfrom = 0
        sampto = 'end'
        channels = 'all'
        record = wfdb.rdrecord(
            file_name,
            sampfrom,
            sampto,
            channels,
            True,
            database,
            True
        )

        record.p_signal = record.p_signal.T
        print 'Record downloaded. \nDownloading annotations...'

        ann = get_annotations()
        print 'Annotations downloaded.'
        record = Record(record, ann)

        if save:
            _save(record, database)
            print 'Record %s saved.' % file_name

        return record

    except requests.exceptions.HTTPError:
        err_msg = 'No file with name: %s in %s found.' % (file_name, database)
        raise NameError, err_msg


def _check_arguments_are_strings(*args):
    for arg in args:
        err_msg = 'Argument \'%s\' must be a string.' % arg
        assert isinstance(arg, str), err_msg


def _if_missing_make_directory_for(database):
    """ Looks for the database folder in the current directory and makes it
    if it does not exist. """

    dir_path = SAVE_PATH + '/' + database
    if not os.path.exists(dir_path):
        print 'Making directory %s...' % dir_path
        os.makedirs(dir_path)


def _is_file_in_directory(name, database):
    file_name = name + '.pkl'
    file_path = SAVE_PATH + '/' + database
    file_list = os.listdir(file_path)

    return file_name in file_list


def _save(record, database, update=True):
    """ Saves instances of the wfdb class Record via pickle to prevent the need
    of the slow process of re-downloading the data from the server every time a
    file is run. Files are saved under the same name as in the database, in
    the folder named after the database in the SAVE_PATH directory.

    Parameters:
        record (wfdb.Record): a physionet record from get_record().

        database (str): name of the database the record is from.

        update (optional bool): used to specify if the Record should be
        downloaded and saved even if the Record already exists locally. """

    _if_missing_make_directory_for(database)
    file_path = (
        SAVE_PATH + '/' + \
        database + '/' + \
        record.record_name + \
        '.pkl'
    )

    if not update and _is_file_in_directory(record.recordname, database):
        return

    print 'Saving file to %s' % file_path
    new_file = open(file_path, 'wb')
    cPickle.dump(record, new_file, protocol=2)


def _load(file_name, database):
    """ loads previously saved files. Files are saved as the same file name as
    on physionet.org in the local database directory if it exists. """

    file_path = SAVE_PATH + file_name
    err_msg = ('File can not be loaded. Directory: %s, does not exist.'
               % file_path)
    assert _is_file_in_directory(file_name, database), err_msg

    print 'Loading %s.pkl...' % file_name
    file_path = SAVE_PATH + '/' + database + '/' + file_name + '.pkl'
    local_file = open(file_path, 'rb')
    return cPickle.load(local_file)


def plot_record(
        record,
        channel=1,
        ignored_annotations=IGNORED_ANNOTATIONS,
        marker='o'
    ):

    """ Plots the signal and annotations from one channel of a record.

    Parameters:
        record (wfdb.Record): a physionet record retrieved via the get_record()
        function.

        channel (optional positive int): the channel whose signal should be
        shown.

        ignored_annotations (optional tuple): Annotations with a symbol in
        the ignored_annotations tuple are not shown on the plot. By default
        ignored_annotations removes all annotations which don't indicate an
        unhealthy event or healthy beat. Printing the get_ann_symbol_table
        function will return a table of annotation symbols and descriptions for
        reference.

        marker (optional symbol): the plot marker. """

    def check_channel_index():
        """ Make sure the channel to plot exists. """

        num_channels = len(record.p_signal)
        err_msg = ('Channel %d out of bounds. Record has only %d channels.'
                   % (channel, num_channels))

        assert channel_index < num_channels, err_msg

        err_msg = 'Channel most be a positive integer.'
        is_positive = channel > 0
        is_int = isinstance(channel, int)
        assert is_positive and is_int, err_msg

    def plot_annotations():
        """ Add record's annotations to the plot. """

        annotations = record.ann.sample
        symbols = record.ann.symbol
        labels = get_symbol_labels()
        handles = []
        legend_keys = []

        for symbol, description in labels.items():
            ann_idx = annotations[symbols == symbol]

            is_anns = ann_idx.sum()
            if is_anns and symbol not in ignored_annotations:
                handle, = plt.plot(time[ann_idx], signal[ann_idx], marker)
                handles.append(handle)
                legend_keys.append(description)

        plt.legend(handles, legend_keys)

    channel_index = channel - 1
    check_channel_index()

    signal = record.p_signal[channel_index]
    time = convert_samples_to_time(record)

    plt.figure('Record %s' % record.record_name)
    plt.plot(time, signal)
    plot_annotations()
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude')


def convert_samples_to_time(record):
    """ Uses the record's sampling frequence to map sample point positions to
    time, in seconds. Returns an array of times. """

    samp_freq = record.fs
    signal = record.p_signal[0]
    time = np.linspace(0, float(len(signal))/samp_freq, len(signal))
    return time


def get_symbol_labels():
    """ Returns a dictionary with the annotation symbols as its keys and the
    symbol's descriptions as its values. """

    table = StringIO(get_ann_symbol_table())
    reader = csv.reader(table)

    labels = {}
    for row in reader:
        symbol = re.findall(r' [^ \d] ', row[0])
        if symbol:
            description = re.findall(r'[A-Z][\S]+[()\/\- A-Za-z]*', row[0])[0]
            labels[symbol[0][1]] = description

    return labels


def get_ann_symbol_table():
    """ Returns a table of annotation labels. """

    return wfdb.io.annotation.ann_label_table


def does_record_contain_ann_symbol(record, symbol):
    """ Checks if record's annotation contains a specific annotation symbol.

    Parameters:
        record (wfdb.Record): a record obtained from get_record().

        symbol (string or list like): an annotation symbol or a list like
        instance of symbols to test.

    Output:
        Returns False if none of the symbols are found or a list of all
        instance of the test symbols in the record. """

    special_chars = ('*', '+', '?', '^', '$', '.')
    if isinstance(symbol, str) and symbol in special_chars:
        symbol = '\\' + symbol

    elif isinstance(symbol, (list, tuple, np.ndarray)):
        check = False
        for sym in symbol:
            check = check or does_record_contain_ann_symbol(record, sym)

        return check

    test_symbol = symbol
    record_symbols = record.ann.symbol
    record_symbols = ''.join(record_symbols)
    list_of_symbols = re.findall(test_symbol, record_symbols)

    return list_of_symbols

rec = get_record('101', 'mitdb')
plot_record(rec)
plt.show()
