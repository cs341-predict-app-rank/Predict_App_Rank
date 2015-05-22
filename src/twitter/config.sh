#local
#spark_main_dir/ec2/
#launch a cluster
./spark-ec2 -k Haoming -i Haoming.pem --region=us-west-2 --zone=us-west-2b \
-s 3 --instance-type=c3.xlarge --hadoop-major-version=2 \
--copy-aws-credentials launch spark-final-test

#login onto a cluster
./spark-ec2 -k Haoming -i Haoming.pem --region=us-west-2 login spark-final-test

#stop a cluster
./spark-ec2 --region=us-west-2 stop spark-final-test

#restart a cluster 
./spark-ec2 -i Haoming.pem --region=us-west-2 --copy-aws-credentials start spark-final-test