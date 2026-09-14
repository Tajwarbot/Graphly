export const AI_TOOLS_DECLARATION = [
    {
        functionDeclarations: [
            {
                name: 'updateExpression',
                description:
                    'Update only an existing layer explicitly requested by the user. Use its layerId from graph context; ask if the target is ambiguous. Never use this to add a new graph.',
                parameters: {
                    type: 'OBJECT',
                    properties: {
                        layerId: { type: 'STRING' },
                        expression: { type: 'STRING' }
                    },
                    required: ['layerId', 'expression']
                }
            },
            {
                name: 'plotFunction',
                description:
                    'Plot an explicit 2D mathematical curve y = f(x), e.g. sin(x), x^2, 1/x, 2x + 1. Must be pure math expression without natural language text.',
                parameters: {
                    type: 'OBJECT',
                    properties: {
                        expression: {
                            type: 'STRING',
                            description:
                                "The mathematical expression in terms of x, e.g. 'x^2', 'sin(x)', '1/x'"
                        }
                    },
                    required: ['expression']
                }
            },
            {
                name: 'plotImplicitEquation',
                description:
                    'Plot a full 2D equation, shaded inequality, or parametric tuple in t. Examples: x^2+y^2=9, y>x, (cos(t),sin(t)){0<t<2*pi}. Supports trailing domain restrictions.',
                parameters: {
                    type: 'OBJECT',
                    properties: {
                        expression: {
                            type: 'STRING',
                            description:
                                "The implicit equation string, e.g. 'x^2 + y^2 = 25'"
                        }
                    },
                    required: ['expression']
                }
            },
            {
                name: 'loadDataTable',
                description:
                    'Load a table of numerical data points into Data Mode for scatter plotting and regression analysis.',
                parameters: {
                    type: 'OBJECT',
                    properties: {
                        name: {
                            type: 'STRING',
                            description: 'Descriptive name for this dataset'
                        },
                        rows: {
                            type: 'ARRAY',
                            description:
                                'Array of point objects containing x and y coordinates',
                            items: {
                                type: 'OBJECT',
                                properties: {
                                    x: {
                                        type: 'NUMBER',
                                        description: 'X coordinate'
                                    },
                                    y: {
                                        type: 'NUMBER',
                                        description: 'Y coordinate'
                                    }
                                },
                                required: ['x', 'y']
                            }
                        }
                    },
                    required: ['rows']
                }
            },
            {
                name: 'switchTo3D',
                description:
                    'Add a new 3D surface. Accepts explicit expressions or complete implicit equations involving x, y, z. Existing surfaces are preserved.',
                parameters: {
                    type: 'OBJECT',
                    properties: {
                        expression: {
                            type: 'STRING',
                            description:
                                "The 3D surface expression in terms of x and y, e.g. 'sin(x) * cos(y)', '(x^2 - y^2) / 4', 'x^2/16 + y^2/9 + z^2/4 = 1'"
                        }
                    },
                    required: ['expression']
                }
            },
            {
                name: 'setViewportBounds',
                description:
                    'Zoom or pan the 2D viewport by setting coordinate domain [xMin, xMax] and range [yMin, yMax].',
                parameters: {
                    type: 'OBJECT',
                    properties: {
                        xMin: { type: 'NUMBER' },
                        xMax: { type: 'NUMBER' },
                        yMin: { type: 'NUMBER' },
                        yMax: { type: 'NUMBER' }
                    },
                    required: ['xMin', 'xMax', 'yMin', 'yMax']
                }
            }
        ]
    }
];

const string = { type: 'STRING' },
    number = { type: 'NUMBER' };
const bounds = {
    type: 'OBJECT',
    properties: Object.fromEntries(
        ['xMin', 'xMax', 'yMin', 'yMax', 'zMin', 'zMax'].map((k) => [k, number])
    ),
    required: ['xMin', 'xMax', 'yMin', 'yMax']
};
const triple = { type: 'ARRAY', items: number };
const camera = {
    type: 'OBJECT',
    properties: { position: triple, target: triple },
    required: ['position', 'target']
};
const style = {
    name: string,
    color: string,
    opacity: number,
    visible: { type: 'BOOLEAN' }
};
AI_TOOLS_DECLARATION[0].functionDeclarations.push(
    {
        name:'setParameter',
        description:'Set an existing numeric parameter by its layer ID. Optionally configure its slider min, max, and positive step. Use IDs from graph context; never guess an ID. Does not create a new parameter.',
        parameters:{type:'OBJECT',properties:{layerId:string,value:number,min:number,max:number,step:number},required:['layerId']}
    },
    {
        name: 'add2DScene',
        description: 'Append a complete 2D artwork atomically as 1–12 named colored layers: explicit/implicit equations, inequalities or parametric tuples. Preserve existing work. Use restrictions to trim strokes and inequalities for filled regions.',
        parameters: {type:'OBJECT',properties:{layers:{type:'ARRAY',items:{type:'OBJECT',properties:{equation:string,...style},required:['equation']}},bounds:{type:'OBJECT',properties:{xMin:number,xMax:number,yMin:number,yMax:number},required:['xMin','xMax','yMin','yMax']}},required:['layers']}
    },
    {
        name: 'addScene',
        description:
            'Append a complete 3D design with 1–12 named, colored surfaces atomically. Derive equations for component shapes. Use implicit equations for closed objects and vertical planes. Supply camera/bounds to frame the design. Never replace unrelated work.',
        parameters: {
            type: 'OBJECT',
            properties: {
                surfaces: {
                    type: 'ARRAY',
                    items: {
                        type: 'OBJECT',
                        properties: { equation: string, ...style },
                        required: ['equation']
                    }
                },
                bounds,
                camera
            },
            required: ['surfaces']
        }
    },
    {
        name: 'set3DView',
        description:
            'Set the 3D sampling domain or camera. Camera coordinates use mathematical x,y,z; z is vertical.',
        parameters: { type: 'OBJECT', properties: { bounds, camera } }
    },
    {
        name: 'setLayerStyle',
        description:
            'Set name, color, opacity, or visibility of one existing layer by its ID.',
        parameters: {
            type: 'OBJECT',
            properties: { layerId: string, ...style },
            required: ['layerId']
        }
    },
    {
        name: 'removeLayer',
        description:
            'Remove only a layer explicitly requested by the user, by its ID.',
        parameters: {
            type: 'OBJECT',
            properties: { layerId: string },
            required: ['layerId']
        }
    }
);

AI_TOOLS_DECLARATION[0].functionDeclarations.push({
    name:'symbolicMath', description:'Simplify or partially differentiate multivariable expressions, integrate polynomials through degree 8 with symbolic coefficients, or solve a numeric-coefficient linear/quadratic equation over the reals. Returns results without modifying the graph. Unsupported problems return errors.',
    parameters:{type:'OBJECT', properties:{operation:{type:'STRING',enum:['simplify','differentiate','integrate','solve']}, expression:{type:'STRING'}, variable:{type:'STRING'}}, required:['operation','expression']}
});
