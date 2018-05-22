from math import sin,cos,asin,sqrt,radians,pi
import torch

AVG_EARTH_RADIUS = 6371  # in km


def compute_sumrr(self, probs, target):
    '''

    :param probs: batch_size*tagset_size
    :param target: batch_size*1
    :return: sum of reciprocal rank
    '''
    prob_target = torch.gather(probs.data, 1, target.data.view(-1, 1))
    comp = torch.gt(probs.data, prob_target)
    rank = torch.add(comp.float().sum(dim=1), 1)
    rr = torch.sum(torch.reciprocal(rank))
    return rr


def compute_dis(predicted, target,thres=150):

    diff = predicted - target
    LAT_DEGREE = 110.6 * 1000
    LNG_DEGREE = 111.3 * 1000
    diff_mt = torch.stack([diff[0, :].mul(LAT_DEGREE), diff[1, :].mul(LNG_DEGREE)], dim=0)
    dis = torch.norm(diff_mt, p=2, dim=0)
    acc = torch.le(dis, thres)
    return dis, acc

def compute_dis_haversine(predicted,target,thres=(1000,5000)):
    '''

    :param predicted: 2*batch
    :param target:
    :return:
    '''
    p = predicted*180/pi
    t = target*180/pi
    # convert to radians
    diff = p-t
    d= torch.sin(diff[0,:]*0.5 ) **2 + torch.cos(p[0,:])*torch.cos(t[0,:])*torch.sin(diff[1,:]*0.5)**2
    h = 2*AVG_EARTH_RADIUS * torch.asin(torch.sqrt(d))
    acc = []
    for t in thres:
        acc.append(torch.le(h,t).data)
    return h,acc

def haversine(point1, point2):
    """ Calculate the great-circle distance between two points on the Earth surface.
    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.
    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))
    :output: Returns the distance between the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.
    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
    return h #in km