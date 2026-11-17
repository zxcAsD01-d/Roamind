"""Shared visible solids and collision geometry, in each furnishing's bounds."""


def part(item, name, center, size, material=None):
    width, height, depth = item['size']
    scale = (width, height, depth)
    bottom = item['center'][1]-height/2
    origin = (item['center'][0], bottom, item['center'][2])
    return {'id': item['id']+':'+name, 'sourceId': item['id'], 'kind': 'furniture',
            'center': [origin[i]+center[i]*scale[i] for i in range(3)],
            'size': [size[i]*scale[i] for i in range(3)],
            'material': material or item['material']}


def legs(item, top, inset=.39):
    return [part(item, f'leg-{x}-{z}', (x, top/2, z), (.075, top, .075), 'oak')
            for x in (-inset, inset) for z in (-inset, inset)]


def table(item):
    return [part(item, 'top', (0, .93, 0), (1, .14, 1))]+legs(item, .86)


def chair(item):
    direction = 1 if item['center'][0] > 0 else -1
    return legs(item, .48)+[
        part(item, 'seat', (0, .52, 0), (1, .08, 1)),
        part(item, 'cushion', (0, .59, 0), (.86, .06, .86), 'fabric'),
        part(item, 'back', (direction*.45, .8, 0), (.1, .4, 1))]


def computer_chair(item):
    return legs(item, .44)+[
        part(item, 'seat', (0, .46, 0), (1, .08, 1), 'metal'),
        part(item, 'cushion', (0, .505, -.025), (.9, .03, .86), 'fabric'),
        part(item, 'back', (0, .77, .45), (.9, .46, .1), 'fabric')]


def sofa(item):
    pieces = legs(item, .12)+[part(item, 'base', (0, .27, 0), (1, .3, 1), 'oak'),
                            part(item, 'back', (-.4, .7, 0), (.2, .6, 1))]
    for z in (-.46, .46):
        pieces.append(part(item, f'arm-{z}', (.05, .6, z), (.9, .55, .08)))
    for i, z in enumerate((-.28, 0, .28)):
        pieces += [part(item, f'seat-{i}', (.08, .48, z), (.78, .17, .265)),
                   part(item, f'back-cushion-{i}', (-.25, .76, z), (.16, .42, .265), 'linen')]
    return pieces


def bed(item):
    return legs(item, .2)+[
        part(item, 'frame', (0, .35, 0), (1, .3, 1), 'oak'),
        part(item, 'mattress', (0, .68, .02), (.95, .36, .94), 'linen'),
        part(item, 'blanket', (0, .88, .15), (.96, .1, .65), 'fabric'),
        part(item, 'headboard', (0, .95, -.46), (1, 1.1, .08), 'oak'),
        *[part(item, f'pillow-{x}', (x, .93, -.3), (.39, .15, .19), 'linen') for x in (-.23, .23)]]


def cabinet(item):
    pieces = [part(item, 'body', (0, .49, 0), (.94, .98, .96)),
              part(item, 'top', (0, .98, 0), (1, .04, 1), 'stone')]
    # Front follows the long axis so cabinets remain usable against the wall.
    along_z = item['size'][2] > item['size'][0]
    for i in range(max(2, round(max(item['size'][0], item['size'][2])/.55))):
        count = max(2, round(max(item['size'][0], item['size'][2])/.55))
        t = -.5+(i+.5)/count
        c = (-.48, .52, t) if along_z else (t, .52, -.48)
        s = (.04, .87, .94/count) if along_z else (.94/count, .87, .04)
        pieces.append(part(item, f'panel-{i}', c, s, 'metal' if item['id'] == 'fridge' else 'oak'))
        c = (-.51, .6, t) if along_z else (t, .6, -.51)
        s = (.025, .025, .09) if along_z else (.09, .025, .025)
        pieces.append(part(item, f'handle-{i}', c, s, 'metal'))
    if item['id'] == 'tv-console':
        pieces += [part(item, 'screen', (0, 2.25, 0), (.08, 2.1, .8), 'screen'),
                   part(item, 'stand', (0, 1.15, 0), (.5, .3, .12), 'metal')]
    return pieces


def bathroom(item):
    if 'shower' in item['id']:
        return [part(item, 'tray', (0, .025, 0), (1, .05, 1), 'ceramic'),
                part(item, 'back', (0, .5, .48), (1, 1, .04), 'glass'),
                part(item, 'side', (.48, .5, 0), (.04, 1, 1), 'glass'),
                part(item, 'rail', (.48, .99, 0), (.025, .02, 1), 'metal'),
                part(item, 'pipe', (.3, .65, .44), (.02, .5, .02), 'metal'),
                part(item, 'shower-head', (.3, .91, .35), (.12, .02, .2), 'metal')]
    if 'toilet' in item['id']:
        return [part(item, 'base', (0, .25, 0), (.55, .5, .65)),
                part(item, 'seat', (0, .59, -.06), (.9, .2, .78)),
                part(item, 'tank', (0, .77, .36), (.85, .46, .25))]
    return [part(item, 'pedestal', (0, .4, .2), (.4, .8, .4)),
            part(item, 'bowl', (0, .87, 0), (1, .18, 1)),
            part(item, 'basin', (0, .97, -.02), (.72, .03, .68), 'stone'),
            part(item, 'faucet', (0, 1.06, .36), (.06, .2, .06), 'metal')]


def detailed_furnishings(items):
    output = []
    for item in items:
        name = item['id']
        if name == 'water-dispenser':
            output.extend(dispenser(item))
        elif name.startswith('sandwich-plate-'):
            output.append(item)
        elif name == 'computer-chair':
            output.extend(computer_chair(item))
        elif name == 'computer-desk' or name.endswith('table'):
            output.extend(table(item))
        elif name in ('computer-tower', 'computer-shelf'):
            output.extend(cabinet(item))
        elif name.startswith('computer-'):
            output.append(item)
        elif name.startswith('chair'):
            output.extend(chair(item))
        elif name == 'sofa':
            output.extend(sofa(item))
        elif name.startswith('bed-'):
            output.extend(bed(item))
        elif any(word in name for word in ('shower', 'toilet', 'basin')):
            output.extend(bathroom(item))
        else:
            output.extend(cabinet(item))
    return output


def dispenser(item):
    return [part(item, 'base', (0, .28, .08), (1, .56, .84)),
            part(item, 'back', (0, .68, .32), (1, .38, .26)),
            part(item, 'tray', (0, .615, -.24), (.92, .01, .52), 'metal'),
            part(item, 'top', (0, .84, -.06), (1, .16, .96)),
            part(item, 'spout', (0, .73, -.37), (.06, .08, .08), 'metal'),
            part(item, 'button', (0, .815, -.55), (.12, .045, .04), 'screen'),
            part(item, 'tank', (0, 1.02, .03), (.7, .35, .68), 'glass')]
