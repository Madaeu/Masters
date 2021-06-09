#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "TemplatedVocabulary.h"
#include <DBoW2.h>

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
    out.resize(plain.rows);

    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

void loadFeatures(std::vector<std::vector<cv::Mat > > &features)
{
    features.clear();

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    cv::String directory = "/home/nicklas/Desktop/00/image_0/*.png";
    std::vector<cv::String> filenames;
    cv::glob(directory, filenames, false);

    std::cout << "Extracting ORB features..." << std::endl;
    features.reserve(filenames.size());
    for(int i = 0; i < (filenames.size()); ++i)
    {

        cv::Mat image = cv::imread(filenames[i], 0);
        cv::Mat mask;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(std::vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }
}

void wait()
{
    std::cout << std::endl << "Press enter to continue" << std::endl;
    getchar();
}


void testVocCreation(const std::vector<std::vector<cv::Mat > > &features)
{
    // branching factor and depth levels
    const int k = 10;
    const int L = 5;
    const DBoW2::WeightingType weight = DBoW2::TF_IDF;
    const DBoW2::ScoringType scoring = DBoW2::L1_NORM;

    OrbVocabulary voc(k, L, weight, scoring);

    std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
    voc.create(features);
    std::cout << "... done!" << std::endl;

    std::cout << "Vocabulary information: " << std::endl
         << voc << std::endl << std::endl;

    // lets do something with this vocabulary
    /*
    std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;
    DBoW2::BowVector v1, v2;
    for(int i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], v1);
        for(int j = 0; j < features.size(); j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
        }
    }

*/

    // save the vocabulary to disk
    std::cout << std::endl << "Saving vocabulary..." << std::endl;
    voc.save("double_voc.yml.gz");
    std::cout << "Done" << std::endl;
}


void testDatabase(const std::vector<std::vector<cv::Mat > > &features)
{
    std::cout << "Creating a small database..." << std::endl;

    // load the vocabulary from disk
    OrbVocabulary voc("KITTIvo02_voc.yml.gz");

    OrbDatabase db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(int i = 0; i < features.size(); i++)
    {
        db.add(features[i]);
    }
    std::cout << "Howdy: " << db.getVocabulary() << std::endl;

    std::cout << "... done!" << std::endl;

    std::cout << "Database information: " << std::endl << db << std::endl;

    // and query the database
    std::cout << "Querying the database: " << std::endl;

    DBoW2::QueryResults ret;
    for(int i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.
        std::cout << "Searching for Image " << i << ". " << ret << std::endl;
    }

    std::cout << std::endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    std::cout << "Saving database..." << std::endl;
    db.save("small_db.yml.gz");
    std::cout << "... done!" << std::endl;
}

int main(){


    std::vector<std::vector<cv::Mat > > features;
    loadFeatures(features);
    testVocCreation(features);
    //Look for img in the database.
 //   cv::Mat img = cv::imread("/home/nicklas/Desktop/2011_09_26_drive_0009_extract/2011_09_26/2011_09_26_drive_0009_extract/image_00/data/0000000300.png");
/*
    //Find descriptors in image
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    features.push_back(std::vector<cv::Mat >());
    changeStructure(descriptors, features.back());

    //Find matching images in the database
    DBoW2::QueryResults ret;
    OrbDatabase db("small_db.yml.gz");
    db.query(features[0], ret, 4); //Features from img, query result, max number of matches
    std::cout << ret << std::endl;
*/
    return 0;
}