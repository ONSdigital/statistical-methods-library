import pytest, os, json, csv


# Import DWT/FXT file to Data Frame
def load_fixture(frameName, version=0):
    if frameName[:4].lower() in ['core', 'look']:
        tablePath = shr.coreTablePath
        suffix = '.dwt'
    elif frameName[:4].lower() in ['char']:
        tablePath = shr.charTablePath
        suffix = '.dwt'
    elif frameName[:4].lower() in ['test']:
        tablePath = shr.testDataPath
        suffix = '.fxt'
    elif frameName[:4].lower() in ['meta']:
        tablePath = shr.metaTablePath
        suffix = '.dwt'
    else:
        if version:
            frameName += '_' + str(version)
        tablePath = os.path.join(shr.basePath, "Components", "tests", "test_tables")
        suffix = '.fxt'
    fileName = str(frameName) + str(suffix)
    dataSource = os.path.join(tablePath, fileName)
    infile = open(dataSource, 'r').read()
    infile = '{' + infile + '}'
    data = json.loads(infile)
    frame = pd.DataFrame(data).transpose()
    #    print('Columns in: ' +frameName)
    #    for colName in frame.columns.tolist():
    #        print(colName)
    #    print('------')
    return frame


# ================== CORE / LOOKUP DATAFRAME FIXTURES ======================
# These never change, so can be safely made session fixtures.

@pytest.fixture(scope="session")
def fx_lookup_biome():
    # set up the dataframe
    tbl.lookupBiome = loadFixture('lookupBiome')
    yield tbl.lookupBiome
    del tbl.lookupBiome