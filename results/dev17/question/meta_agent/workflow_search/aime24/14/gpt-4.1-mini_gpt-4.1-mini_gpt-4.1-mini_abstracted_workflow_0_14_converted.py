async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_instruction_1 = (
        "Sub-task 1: Formally represent the hyperbola and the points A, B, C, D on it, "
        "including the parametrization of points on the hyperbola x^2/20 - y^2/24 = 1. "
        "Avoid attempting to solve equations at this stage; focus solely on expressing the points in a suitable parametric or coordinate form."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, representing hyperbola and points parametrization, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Express the condition that ABCD is a rhombus with diagonals intersecting at the origin, "
        "including the midpoint condition (origin as midpoint of diagonals), equal side lengths, and perpendicularity of diagonals. "
        "Do not yet solve these conditions; only formalize them as vector or coordinate equations."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formalizing rhombus conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Derive the relationships between the vectors representing the diagonals AC and BD based on the midpoint and perpendicularity conditions, "
        "and express the side length equality in terms of these vectors. Avoid numerical solving; focus on symbolic relations and simplifications."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent([taskInfo, thinking2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving vector relations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Combine the hyperbola constraints with the rhombus conditions to reduce the problem to a system of equations involving parameters of vectors AC and BD. "
        "Transform these equations to express the squared lengths of the diagonals and the side length in terms of these parameters."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, combining constraints and reducing system, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Analyze the perpendicularity condition AC · BD = 0 and use it to express one vector parameter in terms of the other, simplifying the system further. "
        "Avoid numerical substitution; focus on algebraic manipulation and parameter reduction."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent([taskInfo, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing perpendicularity and simplifying, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Express the side length equality condition explicitly in terms of the parameters of vectors AC and BD, "
        "and derive an equation relating these parameters. Do not attempt to solve for maximum values yet; focus on obtaining a usable constraint equation."
    )
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent([taskInfo, thinking5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, expressing side length equality, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Using the reduced system of equations, formulate an expression for BD^2 in terms of a single parameter or a pair of parameters, "
        "incorporating all constraints from the hyperbola and rhombus properties. Avoid premature maximization; focus on obtaining a closed-form or parametric expression."
    )
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent([taskInfo, thinking6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating BD^2 expression, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Determine the supremum (greatest lower bound) of BD^2 over all valid parameter values satisfying the constraints, "
        "using calculus or algebraic optimization techniques. Carefully verify that the supremum is approached but not exceeded by any rhombus inscribed on the hyperbola with diagonals intersecting at the origin."
    )
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent([taskInfo, thinking7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, determining supremum of BD^2, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_9 = (
        "Sub-task 9: Interpret the supremum found in Sub-task 8 as the greatest real number less than BD^2 for all such rhombi, "
        "and clearly state the final answer with justification. Avoid introducing new assumptions or conditions; base the conclusion strictly on previous results."
    )
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent([taskInfo, thinking8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, interpreting supremum and finalizing answer, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
