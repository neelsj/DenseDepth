import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def DepthNorm(x, maxDepth):
    xNorm = x
    xNorm[x != 0] = maxDepth / x[x != 0]
    return xNorm


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open(file), dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)


def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)

    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []

        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:, :, 0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:, :, :3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)

    return skimage.util.montage(all_images, multichannel=True, fill=(0, 0, 0))


def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage = display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage * 255))
    im.save(filename)


def load_test_data(datadir='./', test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(datadir + test_data_zip_file)
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')
    return {'rgb': rgb, 'depth': depth, 'crop': crop}


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10


def compute_split_errors(gt, pred, buckets=None):
    if buckets is None:
        buckets = range(11)
    rmses = []

    for idx in range(len(buckets) - 1):
        # range of values to compute error on
        lower = buckets[idx]
        upper = buckets[idx + 1]

        # mask gt and pred according to this range
        gt_mask = ((gt >= lower) & (gt < upper))
        gt_mask = np.invert(gt_mask)
        masked_gt = np.copy(gt)
        np.putmask(masked_gt, gt_mask, 0)
        masked_pred = np.copy(pred)
        np.putmask(masked_pred, gt_mask, 0)
        num_values = np.count_nonzero(masked_gt)

        # calculate RMSE
        rmse = (masked_gt - masked_pred) ** 2
        rmse = np.sum(rmse)
        rmse = rmse / num_values
        rmse = np.sqrt(rmse)

        rmses.append([lower, upper, rmse, num_values])

    return rmses

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False, scale=True, showImages=False, split_errors=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []

    for i in range(N // bs):
        x = rgb[(i) * bs:(i + 1) * bs, :, :, :]

        # Compute results
        true_y = depth[(i) * bs:(i + 1) * bs, :, :]
        pred_y = scale_up(2, predict(model, x / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :, 0])

        if (showImages):

            for b in range(bs):
                plt.subplot(1, 3, 1)
                plt.imshow(x[b, :, :, :])
                plt.show(block=False)

                vmin = np.min(true_y[b, :, :])
                vmax = np.max(true_y[b, :, :])

                plt.subplot(1, 3, 2)
                plt.imshow(true_y[b, :, :], vmin=vmin, vmax=vmax)
                plt.show(block=False)

                plt.subplot(1, 3, 3)
                plt.imshow(pred_y[b, :, :], vmin=vmin, vmax=vmax)
                plt.show(block=False)

                plt.waitforbuttonpress()

        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2,
                               predict(model, x[..., ::-1, :] / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :,
                               0])

        if (scale):
            pred_y *= 10.0
            pred_y_flip *= 10.0

        if (crop is not None):
            # Crop based on Eigen et al. crop
            true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            pred_y_flip = pred_y_flip[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append((0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
            testSetDepths.append(true_y[j])

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    if split_errors:
        e = compute_split_errors(testSetDepths, predictions)

    else:
        e = compute_errors(testSetDepths, predictions)

        if verbose:
            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
            print(
                "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))

    return e


def compare(model, model2, rgb, depth, crop, batch_size=6, verbose=False, scale=True):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []

    for i in range(N // bs):
        x = rgb[(i) * bs:(i + 1) * bs, :, :, :]

        # Compute results
        true_y = depth[(i) * bs:(i + 1) * bs, :, :]
        pred_y = scale_up(2, predict(model, x / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :, 0])
        pred_y2 = scale_up(2, predict(model2, x / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :, 0])

        if (scale):
            pred_y *= 10.0

        for b in range(bs):
            plt.subplot(1, 4, 1)
            plt.imshow(x[b, :, :, :])
            plt.show(block=False)

            vmin = np.min(true_y[b, :, :])
            vmax = np.max(true_y[b, :, :])

            plt.subplot(1, 4, 2)
            plt.imshow(true_y[b, :, :], vmin=vmin, vmax=vmax)
            plt.show(block=False)

            plt.subplot(1, 4, 3)
            plt.imshow(pred_y[b, :, :], vmin=vmin, vmax=vmax)
            plt.show(block=False)

            plt.subplot(1, 4, 4)
            plt.imshow(pred_y2[b, :, :], vmin=vmin, vmax=vmax)
            plt.show(block=False)

            plt.waitforbuttonpress()

    return e
