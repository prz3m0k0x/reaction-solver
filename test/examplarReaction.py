        
if __name__ == "__main__":

    UNIVERSALGASCONSTANT = 8.31446261815324
    cp_o2_coeffs = [3.53]
    cp_so2_coeffs = [3.53]
    cp_so3_coeffs = [3.53]
    cp_n2_coeffs = [3.53]

    SO2 = Specie(
        name="so2",
        molarMass= 0.64,
        heatCapacityModel="polynomial",
        enthalpyFormation=-296861.220,
        entropyFormation=248.100,   
        heatCapacityCoefficients=cp_so2_coeffs
    )

    O2 = Specie(
        name="o2",
        molarMass=.32,
        heatCapacityModel="polynomial",
        enthalpyFormation= 0.0,
        entropyFormation=205.000,    
        heatCapacityCoefficients=cp_o2_coeffs
    )

    N2 = Specie(
        name="n2",
        molarMass=.28,
        heatCapacityModel="polynomial",
        enthalpyFormation= 0.0,
        entropyFormation=191.500,    
        heatCapacityCoefficients=cp_n2_coeffs
    )

    SO3 = Specie(
        name="so3",
        molarMass=.80,
        heatCapacityModel="polynomial",
        enthalpyFormation=-395782.300,
        entropyFormation=256.600,    
        heatCapacityCoefficients=cp_so3_coeffs
    )
    species = [SO2, O2, SO3, N2]

    reaction = Reaction(
        name="so2conversion",
        stochiometricCoefficients=np.array([-1.0, -0.5, 1.0, 0.0]),
        speciesExponent=np.array([1.0, 0.5, 0.0, 0.0]),
        reversedSpecieExponent=np.array([0.0, 0.0, 1.0, 0.0]),
        isReversible=True,
        ahrreniusPreExponent= 50e9,
        ahrreniusActivationEnergy= 165000.,
        species=species
    )

    z0 = Zone(length=1.0, zoneType="reaction")
    z0.zoneAssign(heating=False, reaction=True)
    z0.zoneAssignHeating(0.0)

    z1 = Zone(length=1.0, zoneType="heating")
    z1.zoneAssign(heating=True, reaction=False)
    z1.zoneAssignHeating(-50000.0)

    z2 = Zone(length=1.0, zoneType="reaction")
    z2.zoneAssign(heating=False, reaction=True)
    z2.zoneAssignHeating(0.0)


    Y_so2 = 0.08
    Y_so3 = 1e-6
    Y_o2 = 0.21 * (1 - Y_so2)
    Y_n2 = 1 - Y_so2 - Y_o2 - Y_so3
    
    domain = domainSetup(
        diameter=2.5,
        inletMassFractions=np.array([Y_so2, Y_o2, Y_so3, Y_n2])
    )
    mesh = Mesh(domain=domain, zoneList=[z0, z1, z2], sizing=0.005)
    mesh.meshCreate()

    Yso2_field = scalarField("Y_so2")
    Yo2_field = scalarField("Y_o2")
    Yso3_field = scalarField("Y_so3")
    Yn2_field = scalarField("Y_n2")

    Yso2_field.fieldInitialize(mesh)
    Yo2_field.fieldInitialize(mesh)
    Yso3_field.fieldInitialize(mesh)
    Yn2_field.fieldInitialize(mesh)
    specieFields = [Yso2_field, Yo2_field, Yso3_field, Yn2_field]
    
    inlet = Inlet(0, 2.5, 690, [Y_so2, Y_o2, Y_so3, Y_n2])
    mixture = Mixture(
        densityModel="ideal-incompressible-gas",
        densityValue=0.457,
        species=species
    )

    sol = solver(mesh=mesh, mixture=mixture, reaction=reaction, specieFields=specieFields, inlet=inlet)
    sol.initializeCase()

    sol.steadyState(max_iter=450, relaxationFactorSpecie=0.2, relaxationFactorTemperature=0.15, convergenceCriteria=1e-5)
    
    plotter = ReactorPlotter(sol)
    plotter.plot_all()

    z = sol.mesh.cell_centers
    plt.figure(figsize=(10, 4))
    plt.plot(z, sol.heatSources, label="Q_rxn + Q_ext", color='purple')
    plt.axvspan(0.0, 1.0, color="red", alpha=0.1, label="Rxn zone 1")
    plt.axvspan(1.0, 2.0, color="blue", alpha=0.1, label="Cooling")
    plt.axvspan(2.0, 3.0, color="red", alpha=0.1, label="Rxn zone 2")
    plt.xlabel("Axial position z [m]")
    plt.ylabel("Heat Source [W/m³]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()