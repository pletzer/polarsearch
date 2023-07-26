import numpy as np
import defopt
import vtk
from find_param import findParam


def collect_iteration_points(rho_0: float=0.5, the_0: float=np.pi/3, x_target: float=0.7, y_target: float=0.3, num_iters: int=0) -> np.array:

    xi = np.array([rho_0, the_0])
    target = np.array([x_target, y_target])
    res = np.empty((num_iters + 2, 3), float)
    x = xi[0]*np.cos(xi[1])
    y = xi[0]*np.sin(xi[1])
    res[0, :] = x, y, 0.
    for i in range(num_iters):
        r0, t0 = xi
        r02 = r0*r0
        x0, y0 = r0*np.cos(t0), r0*np.sin(t0)
        dx, dy = x_target - x0, y_target - y0
        xi += np.array([(+ x0*dx + y0*dy)/r0,
                        (- y0*dx + x0*dy)/r02])
        x = xi[0]*np.cos(xi[1])
        y = xi[0]*np.sin(xi[1])
        res[i + 1, :] = x, y, 0.
    res[num_iters + 1, :] = x_target, y_target, 0.0
    return res     


def main(*, rho_0: float=0.5, the_0: float=0., tol: float=1.e-4, nr: int=11, nt: int=21):

    num_iters = vtk.vtkFloatArray()
    num_iters.SetNumberOfComponents(1)
    num_iters.SetNumberOfTuples(nr*nt)
    num_iters.SetName('num_iters')

    point_array = vtk.vtkFloatArray()
    point_array.SetNumberOfComponents(3)
    point_array.SetNumberOfTuples(nr*nt)

    rho_min = 0.1
    max_val = 0
    index = 0
    worst_target = np.zeros((3,), np.float64)
    for i in range(nr):
        rho = rho_min + (1. - rho_min)*i/(nr - 1)
        for j in range(nt):
            the = 0. + 2*np.pi*j/(nt - 1)
            x = rho*np.cos(the)
            y = rho*np.sin(the)
            val = find(rho_0, the_0, x, y, tol)
            num_iters.SetTuple(index, (val,))
            if val > max_val:
                worst_target[:] = x, y, 0.
            max_val = max(val, max_val)
            point_array.SetTuple(index, (x, y, 0.01*val))
            index += 1

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nr*nt)
    points.SetData(point_array)

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nt, nr, 1)
    grid.SetPoints(points)
    grid.GetPointData().SetScalars(num_iters)

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(grid)

    lut = vtk.vtkLookupTable()
    print(f'max_val = {max_val} worst target = {worst_target}')
    lut.SetTableRange(0., max_val)
    lut.SetNumberOfTableValues(128)
    for i in range(128):
        x = i/(128 - 1)
        r = 1.0*(1 - x)**2
        g = 215*(1 - x)**2/255 + x**2
        b = 0.2*x**2
        lut.SetTableValue(i, r, g, b)
    #lut.SetHueRange(0, 0.667)
    #lut.UsingLogScale()
    lut.Build()
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(1)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    sphere_source = vtk.vtkSphereSource()
    x0 = rho_0 * np.cos(the_0)
    y0 = rho_0 * np.sin(the_0)
    sphere_source.SetCenter((x0, y0, 0.))
    sphere_source.SetRadius(0.05)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.GetProperty().SetColor(0., 1, 0)

    # worst case

    traj_xyz = collect_iteration_points(rho_0, the_0, worst_target[0], worst_target[1], max_val)
    print(traj_xyz)
    narcs = traj_xyz.shape[1] - 1
    zhat = np.array([0., 0., 1.])
    traj_pipeline = []    
    for i in range(narcs):
        s = vtk.vtkArcSource()
        p0, p1 = traj_xyz[i, :], traj_xyz[i + 1, :]
        dp = p1 - p0
        dist = np.sqrt(dp.dot(dp))
        p2 = 0.5*(p0 + p1) + dist * zhat

        s.SetPoint1(p0)
        s.SetPoint2(p1)
        s.SetCenter(p2)
        s.SetResolution(10)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(s.GetOutputPort())
        a = vtk.vtkActor()
        a.GetProperty().SetColor(1, 0, 0)
        a.SetMapper(m)
        traj_pipeline.append((s, m, a))

        s2 = vtk.vtkSphereSource()
        s2.SetCenter(p1)
        s2.SetRadius(0.02)
        m2 = vtk.vtkPolyDataMapper()
        m2.SetInputConnection(s2.GetOutputPort())
        a2 = vtk.vtkActor()
        a2.GetProperty().SetColor(1, 0, 0)
        a2.SetMapper(m2)
        traj_pipeline.append((s2, m2, a2))

    # traj_pipeline = []
    # for i in range(traj_xyz.shape[1]):
    #     s = vtk.vtkSphereSource()
    #     m = vtk.vtkPolyDataMapper()
    #     m.SetInputConnection(s.GetOutputPort())
    #     a = vtk.vtkActor()
    #     a.SetMapper(m)
    #     s.SetCenter(traj_xyz[i, :])
    #     s.SetRadius(0.04)
    #     traj_pipeline.append((s, m, a))


    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor)
    renderer.AddActor(sphere_actor)
    for p in traj_pipeline:
        renderer.AddActor(p[2])

    renderWindow.SetSize(640, 480)
    renderWindow.Render()
    renderWindowInteractor.Start()

            
    #print(num_iters)
    #plt.imshow(num_iters)
    #plt.show()


if __name__ == '__main__':
    defopt.run(main)