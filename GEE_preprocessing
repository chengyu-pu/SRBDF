// Define region of interest
var roi = ee.FeatureCollection("projects/LarsenC");
// Styling for ROI display
Map.centerObject(roi, 10);
var styling = {color: "red", fillColor: "00000000"};
Map.addLayer(roi.style(styling), {}, "ROI");
// Load Sentinel-1 collection and filter by ROI, date, polarization, and mode
var sentinel1Collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate('2020-10-20', '2021-04-01')
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HH'))
  .filter(ee.Filter.eq('instrumentMode', 'EW'));
// Separate ascending and descending collections based on 'orbitProperties_pass'
var ascendingCollection = sentinel1Collection.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')); // Ascending orbit
var descendingCollection = sentinel1Collection.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')); // Descending orbit
// Get the first image from each collection
var ascendingImage = ascendingCollection.first();  
var descendingImage = descendingCollection.first();  
// Function to convert dB to linear power
function dbToPower(dbImage) {
  return ee.Image(10).pow(dbImage.divide(10));
}
// Function to convert linear power to dB
function powerToDb(powerImage) {
  return ee.Image(10).multiply(powerImage.log10());
}
// Refined Lee Filter for speckle noise reduction
function refinedLeeFilter(image) {
    var imgBand = image.select(['HH']);
    imgBand = dbToPower(imgBand);
    // Set up 3x3 kernels 
    var weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
    var kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, false);
    var mean3 = imgBand.reduceNeighborhood(ee.Reducer.mean(), kernel3);
    var variance3 = imgBand.reduceNeighborhood(ee.Reducer.variance(), kernel3);
     // Sample a 7x7 window for gradient/direction analysis
    var sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);
    var sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, false);
    // Compute mean and variance for sampled windows
    var sample_mean = mean3.neighborhoodToBands(sample_kernel); 
    var sample_var = variance3.neighborhoodToBands(sample_kernel);
    // Determine the 4 gradients for the sampled windows
    var gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());
    // Find maximum gradient per pixel
    var max_gradient = gradients.reduce(ee.Reducer.max());
    // Create a mask for band pixels that are the maximum gradient
    var gradmask = gradients.eq(max_gradient);
    // duplicate gradmask bands: each gradient represents 2 directions
    gradmask = gradmask.addBands(gradmask);
    // Determine directions
    var directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
    directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
    directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
    directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
    directions = directions.addBands(directions.select(0).not().multiply(5));
    directions = directions.addBands(directions.select(1).not().multiply(6));
    directions = directions.addBands(directions.select(2).not().multiply(7));
    directions = directions.addBands(directions.select(3).not().multiply(8));
    directions = directions.updateMask(gradmask);
    directions = directions.reduce(ee.Reducer.sum());  
    // Local noise estimation
    var sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
    var sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);
    // Set up the 7*7 kernels for directional statistics
    var rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));
    var diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], 
      [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);
    var rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, false);
    var diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, false);
    // Create stacks for mean and variance using the original kernels. Mask with relevant direction.
    var dir_mean = imgBand.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
    var dir_var = imgBand.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));
    dir_mean = dir_mean.addBands(imgBand.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
    dir_var = dir_var.addBands(imgBand.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));
    // Add the bands for rotated kernels
    for (var i=1; i<4; i++) {
      dir_mean = dir_mean.addBands(imgBand.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
      dir_var = dir_var.addBands(imgBand.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
      dir_mean = dir_mean.addBands(imgBand.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
      dir_var = dir_var.addBands(imgBand.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
    }
    dir_mean = dir_mean.reduce(ee.Reducer.sum());
    dir_var = dir_var.reduce(ee.Reducer.sum());
    // Refined Lee filtered value
    var varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0));
    var b = varX.divide(dir_var);
    var refinedLeeFilter = dir_mean.add(b.multiply(imgBand.subtract(dir_mean)))
    .arrayProject([0])
    .arrayFlatten([['sum']])
    .float();
    return image.addBands(refinedLeeFilter.rename('HH_filtered'));
}
// Normalize HH band to a reference incidence angle
function anglenor(image) {
  var angle = image.select('angle'); 
  var HH = image.select('HH_filtered');
  var cos_angle = angle.divide(180).multiply(Math.PI).cos().pow(2); 
  var reference_angle = ee.Number(40).divide(180).multiply(Math.PI).cos().pow(2); 
  var HH_ref = HH.multiply(reference_angle).divide(cos_angle); 
  var HH_ref_db = powerToDb(HH_ref).rename('HH_normalized');
  var imageWithNormalizedBand = image.addBands(HH_ref_db);
  return imageWithNormalizedBand;
}
// Preprocess image collection: speckle filtering + incidence angle normalization
function preprocessCollection(collection) {
  var processedCollection = collection.map(refinedLeeFilter).map(anglenor);
  return processedCollection;
}
// Process the ascending and descending collections
var processedAscending = preprocessCollection(ascendingCollection);
var processedDescending = preprocessCollection(descendingCollection);
// Function to get band names for each image in the collection
function getBandNames(image) {
  var bandNames = image.bandNames();
  var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
  return ee.Feature(null, {
    'date': date,
    'bandNames': bandNames
  });
}
var bandNamesAscending = processedAscending.map(getBandNames);
var bandNamesDescending = processedDescending.map(getBandNames);
// Print the results
print('Band names for each ascending image:', bandNamesAscending);
print('Band names for each descending image:', bandNamesDescending);
print('Number of ascending images: ', processedAscending.size());
print('Number of descending images: ', processedDescending.size());
// Select first image for visualization
var firstAscending = processedAscending.first();
var firstDescending = processedDescending.first();
var originalHH_Ascending = firstAscending.select('HH');
var originalHH_Descending = firstDescending.select('HH');
var normalizedHH_Ascending = firstAscending.select('HH_normalized');
var normalizedHH_Descending = firstDescending.select('HH_normalized');
// Visualization parameters
var visParams = {
  min: -35,  
  max: 5,
  palette: ['black', 'white']  
};
Map.addLayer(originalHH_Ascending, visParams, 'Original HH Ascending');
Map.addLayer(normalizedHH_Ascending, visParams, 'Normalized HH Ascending');
Map.addLayer(originalHH_Descending, visParams, 'Original HH Descending');
Map.addLayer(normalizedHH_Descending, visParams, 'Normalized HH Descending');
// Compute min/max stats for first image
var statsParams = {
  reducer: ee.Reducer.minMax(),
  geometry: roi.geometry(),
  scale: 40, 
  maxPixels: 1e13
};
var ascendingStats = firstAscending.select('HH_normalized').reduceRegion(statsParams);
var descendingStats = firstDescending.select('HH_normalized').reduceRegion(statsParams);
print('First ascending HH normalized backscatter: ', ascendingStats);
print('First descending HH normalized backscatter: ', descendingStats);
// Function to export each image in a collection to Google Drive
function exportImageCollection(imgCol, roi, scale, orbitType) {
  var indexList = imgCol.reduceColumns(ee.Reducer.toList(), ["system:index"]).get("list");
  indexList.evaluate(function(indexs) {
    for (var i = 0; i < indexs.length; i++) {
      var image = imgCol.filter(ee.Filter.eq("system:index", indexs[i])).first();
      image = image.select('HH_normalized').toFloat();
      Export.image.toDrive({
        image: image.clip(roi), // Clip the image to the region of interest
        description: orbitType + "_" + indexs[i], // Add orbit type to description
        fileNamePrefix: orbitType + "_" + indexs[i], // Set file name prefix with orbit type
        region: roi.geometry(), // Specify the region to export
        scale: scale, // Define the scale/resolution for export
        crs: "EPSG:3031", // Set the coordinate reference system
        maxPixels: 1e13 // Allow a very large number of pixels
      });
    }
  });
}
// Export parameters
var exportScale = 40; 
// Export Ascending and Descending collections separately
exportImageCollection(processedAscending, roi, exportScale, "Ascending");
exportImageCollection(processedDescending, roi, exportScale, "Descending");
