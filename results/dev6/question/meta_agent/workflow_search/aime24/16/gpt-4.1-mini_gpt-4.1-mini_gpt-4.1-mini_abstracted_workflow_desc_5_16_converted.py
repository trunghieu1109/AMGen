async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Extract and formalize all given information and constraints from the problem statement. "
        "Identify the known elements: triangle ABC with circumcenter O and incenter I, circumradius R=13, inradius r=6, "
        "and the perpendicularity condition IA perpendicular to OI. Represent these elements in a coordinate or vector framework or using geometric properties to prepare for further analysis. "
        "Avoid assumptions beyond standard Euclidean geometry."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_1: {subtask_desc1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting and formalizing given data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the formalized data from Sub-task 1, derive fundamental geometric relations. "
        "Express the positions of points O, I, and A in coordinate or vector form, incorporate the known radii R=13 and r=6, "
        "and use the perpendicularity condition IA perpendicular to OI to relate these points. Derive the metric relation IA^2 = R^2 - OI^2 and compute IA in terms of R and r. "
        "Avoid introducing assumptions about angle A or the triangle's shape at this stage. The goal is to establish valid algebraic and geometric relations that will support angle and side length inference."
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
    print(f"Logging before subtask_2: {subtask_desc2}")
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, deriving composite relations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, 
                                                    "Sub-task 2: Synthesize and choose the most consistent and correct geometric relations derived from previous outputs.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_3a = (
        "Sub-task 3a: Use the incenter-distance formula IA = r / sin(A/2) together with the previously derived value of IA to solve for sin(A/2) and thus determine the measure of angle A. "
        "This step must be done rigorously without assuming angle A is 90° or any other specific value. Carefully derive angle A from the given radii and perpendicularity condition, ensuring no unjustified assumptions are made. "
        "Document all algebraic steps and trigonometric manipulations clearly. Reflect on previous outputs and refine the derivation accordingly."
    )
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_3a = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before subtask_3a: {subtask_desc3a}")
    thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, rigorously deriving angle A, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_3a([taskInfo, thinking3a, answer3a], 
                                               "Please review and provide limitations or errors in the above derivation of angle A. If correct, output exactly 'True' in 'correct'.", 
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining angle A derivation, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    debate_instruction_3b = (
        "Sub-task 3b: Verification and Debate: Critically analyze the inference about angle A obtained in subtask_3a. "
        "Multiple agents should debate the validity of the derived angle A, challenge any implicit assumptions, and explore alternative geometric configurations or counterexamples that satisfy the given conditions. "
        "Explicitly forbid accepting angle A = 90° without proof and confirm or refute the angle measure rigorously. The goal is to prevent propagation of errors and ensure the correctness of the key geometric inference before proceeding."
    )
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking_3b = [[] for _ in range(N_max_3b)]
    all_answer_3b = [[] for _ in range(N_max_3b)]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": debate_instruction_3b,
        "context": ["user query", thinking3a.content, answer3a.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before subtask_3b: {subtask_desc3b}")
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking3b, answer3b = await agent([taskInfo, thinking3a, answer3a], debate_instruction_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking3a, answer3a] + all_thinking_3b[r-1] + all_answer_3b[r-1]
                thinking3b, answer3b = await agent(input_infos_3b, debate_instruction_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating angle A inference, thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking_3b[r].append(thinking3b)
            all_answer_3b[r].append(answer3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo] + all_thinking_3b[-1] + all_answer_3b[-1], 
                                                        "Sub-task 3b: Synthesize and finalize the verification of angle A inference.", 
                                                        is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing debate on angle A, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Symbolically or numerically solve the system defined by the vector and circle constraints, including the perpendicularity condition IA perpendicular to OI, "
        "the known radii R=13 and r=6, and the derived angle A, to find the actual side lengths AB and AC or their trigonometric equivalents. "
        "Avoid assumptions about the triangle's shape and rely on solving the system rigorously. Provide explicit expressions or numeric approximations for AB and AC or their product, ensuring consistency with all prior results."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3b.content, answer3b.content, thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_4: {subtask_desc4}")
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3b, answer3b, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, solving system for side lengths, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Calculate the product AB * AC using the inferred parameters from subtask_4. "
        "Use appropriate formulas such as AB = 2R sin C and AC = 2R sin B, or other valid geometric relations, substituting the values or expressions found previously. "
        "Perform algebraic simplifications and numeric computations carefully to obtain a precise numeric value for AB * AC. "
        "Avoid skipping verification of intermediate results to ensure accuracy."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_5: {subtask_desc5}")
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating product AB*AC, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Verification and Cross-Check: Confirm the computed value of AB * AC by cross-checking with alternative geometric relations or by verifying consistency with the given constraints, "
        "including the perpendicularity condition IA perpendicular to OI, the known radii R and r, and the derived angle A. Recompute IA via r / sin(A/2) and check if it matches the metric relation IA^2 = R^2 - OI^2. "
        "If discrepancies arise, revisit previous subtasks to identify and correct errors. Provide a final answer with a justification summary and verification results."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content, answer5.content, thinking3b.content, answer3b.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before subtask_6: {subtask_desc6}")
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5, thinking3b, answer3b], cot_instruction_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, verifying product AB*AC, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
