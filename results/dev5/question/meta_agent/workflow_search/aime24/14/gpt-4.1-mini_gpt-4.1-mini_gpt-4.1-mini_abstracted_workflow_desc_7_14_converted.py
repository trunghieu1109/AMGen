async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Derive algebraic representations for points A, B, C, and D on the hyperbola x^2/20 - y^2/24 = 1. "
        "Express these points in terms of parameters reflecting symmetry and midpoint conditions (origin as midpoint of diagonals). "
        "Formulate vector expressions for diagonals AC and BD, ensuring A and C, and B and D are symmetric about the origin. "
        "Avoid assuming any specific orientation prematurely."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving algebraic representations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the algebraic representations from Sub-task 1, formulate the geometric constraints of the rhombus: equal side lengths, perpendicular diagonals, and diagonals bisecting each other at the origin. "
        "Translate these into algebraic equations involving the coordinates or parameters of points A, B, C, and D. Ensure the conditions are consistent with the hyperbola constraints. "
        "Avoid conflating the labeling or ordering of points without justification."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, formulating rhombus constraints, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, 
                                                    "Sub-task 2: Synthesize and choose the most consistent rhombus constraints formulation.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Identify and parametrize all possible pairs of points (A, C) and (B, D) on the hyperbola that satisfy the midpoint and symmetry conditions. "
        "Verify these pairs satisfy the rhombus conditions derived previously, including equal side lengths and perpendicular diagonals. "
        "Enumerate or characterize the solution set, considering the hyperbola's parametric form and the constraints. "
        "Avoid overlooking implicit domain restrictions or degenerate cases. Provide explicit parameter domains and discuss their geometric meaning."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, parametrizing points and verifying rhombus conditions, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, 
                                                    "Sub-task 3: Synthesize and choose the most consistent parametrization and verification.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    cot_sc_instruction_4 = (
        "Sub-task 4: Analyze the relationships between the parameters of points A, B, C, and D to express the squared length BD^2 in terms of these parameters. "
        "Verify the domain and range of these parameters to ensure all points lie on the hyperbola and satisfy the rhombus conditions. "
        "Avoid overlooking any implicit constraints from the hyperbola or rhombus properties."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, expressing BD^2 and verifying parameters, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, 
                                                    "Sub-task 4: Synthesize and choose the most consistent expression for BD^2.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    debate_instruction_5 = (
        "Sub-task 5: Carefully interpret the problem statement to clarify whether the goal is to find the supremum (least upper bound) or infimum (greatest lower bound) of BD^2 over all valid rhombi. "
        "Analyze both minimal and maximal values of BD^2 under the given constraints. "
        "Formulate the optimization problem accordingly, considering the full parameter domain. "
        "Use calculus or algebraic methods to find critical points and analyze boundary behavior. "
        "Explicitly avoid assuming the maximum or minimum is attained without verification. "
        "Document the reasoning process and intermediate conclusions clearly. "
        "Include the exact problem wording and emphasize interpretation of 'the greatest real number less than BD^2 for all such rhombi'."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing BD^2 extremal values, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                    "Sub-task 5: Synthesize and choose the most consistent interpretation and optimization result for BD^2.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    reflexion_instruction_6 = (
        "Sub-task 6: Based on the analysis in subtask 5, verify which extremal value of BD^2 corresponds to the problem's request for the greatest real number less than BD^2 for all such rhombi. "
        "Confirm that the final answer aligns with the problem's intent, not just the mathematical extremum. "
        "Provide a final answer for this greatest lower bound (infimum) of BD^2, accompanied by a detailed justification. "
        "Reflect on the entire reasoning process, check for contradictions or overlooked cases, and synthesize the findings into a coherent conclusion."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": reflexion_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, verifying final BD^2 bound, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], 
                                                "Please review and provide the limitations of the final solution. If correct, output exactly 'True' in 'correct'.", 
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final verification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
