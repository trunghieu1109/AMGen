async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    reflexion_cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    reflexion_critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    instruction_1 = (
        "Sub-task 1: Clarify and formalize the geometric setup of the problem. Explicitly state the tangency conditions: only the first circle in the chain is tangent to side AB, only the last circle is tangent to side BC, and all interior circles are tangent only to their immediate neighbors. "
        "Define variables for the angle at vertex B (denote as theta), the radius r of each circle, and the number n of circles in the chain. Introduce the concept of the wedge formed by sides AB and BC and note that the chain of circles lies inside this wedge. Emphasize that the centers of the circles do not lie on the angle bisector but on a curve equidistant from the sides, and that the chain offset (the total distance between the first and last circle centers projected along a suitable axis) must be carefully defined. Avoid making unjustified assumptions about the locus of centers or the triangle dimensions beyond what is given.")

    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }

    print(f"Starting {subtask_desc_1['subtask_id']}: {instruction_1}")
    thinking_1, answer_1 = await cot_agent([taskInfo], instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, clarifying geometric setup, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    instruction_2 = (
        "Sub-task 2: Reformulate the geometric model to simplify analysis. Use geometric transformations such as unfolding or reflecting the wedge formed by sides AB and BC about the angle bisector to transform the problem into an equivalent one with two parallel lines separated by a distance related to theta and r. "
        "In this transformed setup, the chain of tangent circles becomes a linear chain between parallel lines, where each circle is tangent to both lines and its neighbors. Derive the correct expression for the total offset (distance between centers of the first and last circles) in terms of the number of circles n, radius r, and the angle theta. Avoid using the invalid formula from previous attempts. Carefully justify each step and ensure the model matches the original tangency conditions. This subtask sets the foundation for accurate formula derivation.")

    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT"
    }

    print(f"Starting {subtask_desc_2['subtask_id']}: {instruction_2}")
    thinking_2, answer_2 = await cot_agent([taskInfo, thinking_1, answer_1], instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, reformulating geometric model, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    instruction_3 = (
        "Sub-task 3: Derive the formula relating the radius r of the tangent circles, the angle theta at vertex B, and the number n of circles in the chain, based on the transformed geometric model. "
        "Express the total offset (length along the chain) as a function of these parameters. Use the fact that the first and last circles are tangent to sides AB and BC respectively, and that the chain fits exactly inside the wedge. Avoid skipping steps and ensure clarity in the derivation. Prepare the formula to be used with the two given configurations (8 circles of radius 34 and 2024 circles of radius 1)."
    )

    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": instruction_3,
        "context": ["user query", thinking_2.content, answer_2.content],
        "agent_collaboration": "CoT"
    }

    print(f"Starting {subtask_desc_3['subtask_id']}: {instruction_3}")
    thinking_3, answer_3 = await cot_agent([taskInfo, thinking_2, answer_2], instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving formula, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    instruction_4 = (
        "Sub-task 4: Use the two given configurations (8 circles of radius 34 and 2024 circles of radius 1) to set up a system of equations based on the formula derived in subtask_3. "
        "These equations relate the angle theta and the inradius r_in of triangle ABC. Carefully handle the scaling and ensure the equations are consistent with the problem statement and the geometric model. Avoid making unjustified assumptions about the triangle's side lengths or the relationship between the chain radius and the inradius. Explicitly state any assumptions made and verify their validity."
    )

    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": instruction_4,
        "context": ["user query", thinking_3.content, answer_3.content],
        "agent_collaboration": "CoT"
    }

    print(f"Starting {subtask_desc_4['subtask_id']}: {instruction_4}")
    thinking_4, answer_4 = await cot_agent([taskInfo, thinking_3, answer_3], instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, setting up equations, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    instruction_5 = (
        "Sub-task 5: Solve the system of equations obtained in stage_1 to find the angle theta at vertex B and the inradius r_in of triangle ABC. "
        "Simplify the expressions to obtain the inradius as a reduced fraction m/n with relatively prime positive integers m and n. Carefully verify each algebraic manipulation to avoid errors. Cross-validate the derived inradius formula with known geometric properties of the inradius and the angle bisector. Avoid conflating the chain circle radius with the inradius and ensure the geometric meaning of each variable is clear."
    )

    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": instruction_5,
        "context": ["user query", thinking_4.content, answer_4.content],
        "agent_collaboration": "SC_CoT"
    }

    print(f"Starting {subtask_desc_5['subtask_id']}: {instruction_5}")

    possible_answers_5 = []
    possible_thinkings_5 = []
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_sc_agents[i]([taskInfo, thinking_4, answer_4], instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, solving system, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_5.append(answer_i)
        possible_thinkings_5.append(thinking_i)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5(
        [taskInfo] + possible_answers_5 + possible_thinkings_5,
        "Sub-task 5: Synthesize and choose the most consistent and correct solution for the system of equations.",
        is_sub_task=True)

    agents.append(f"Final Decision agent, synthesizing solutions, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    instruction_6 = (
        "Sub-task 6: Conduct a reflexion and debate phase to compare alternative formulas for the inradius derived from different geometric approaches (e.g., direct angle bisector relations, chain offset formulas, known inradius formulas involving the angle at B). "
        "Justify and reconcile differences to ensure the correctness of the final formula. This step is crucial to avoid conceptual errors and to confirm that the solution aligns with geometric intuition and known theorems. Document the reasoning and conclusions clearly."
    )

    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": instruction_6,
        "context": ["user query", thinking_5.content, answer_5.content],
        "agent_collaboration": "Debate"
    }

    print(f"Starting {subtask_desc_6['subtask_id']}: {instruction_6}")

    all_thinking_6 = [[] for _ in range(self.max_round)]
    all_answer_6 = [[] for _ in range(self.max_round)]

    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking_6, answer_6 = await agent([taskInfo, thinking_5, answer_5], instruction_6, r, is_sub_task=True)
            else:
                inputs_6 = [taskInfo, thinking_5, answer_5] + all_thinking_6[r-1] + all_answer_6[r-1]
                thinking_6, answer_6 = await agent(inputs_6, instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_6.content}; answer: {answer_6.content}")
            all_thinking_6[r].append(thinking_6)
            all_answer_6[r].append(answer_6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_6_final, answer_6_final = await final_decision_agent_6([taskInfo] + all_thinking_6[-1] + all_answer_6[-1],
                                                                   "Sub-task 6: Synthesize debate results and provide final reconciled formula and reasoning.",
                                                                   is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing debate results, thinking: {thinking_6_final.content}; answer: {answer_6_final.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6_final.content}; answer - {answer_6_final.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6_final, "answer": answer_6_final}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    instruction_7 = (
        "Sub-task 7: Verify the solution by checking the consistency of the inradius with the original problem conditions, including the arrangement of circles and the two given configurations. "
        "Confirm that the fraction m/n is in simplest form and compute m + n. Provide the final answer along with a brief explanation of the verification process. If inconsistencies are found, trigger a reflexion phase to revisit previous steps."
    )

    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": instruction_7,
        "context": ["user query", thinking_6_final.content, answer_6_final.content],
        "agent_collaboration": "Reflexion"
    }

    print(f"Starting {subtask_desc_7['subtask_id']}: {instruction_7}")

    cot_inputs_7 = [taskInfo, thinking_6_final, answer_6_final]

    thinking_7, answer_7 = await reflexion_cot_agent(cot_inputs_7, instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_cot_agent.id}, verifying solution, thinking: {thinking_7.content}; answer: {answer_7.content}")

    for i in range(self.max_round):
        feedback_7, correct_7 = await reflexion_critic_agent([taskInfo, thinking_7, answer_7],
                                                           "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                                           i, is_sub_task=True)
        agents.append(f"Critic agent {reflexion_critic_agent.id}, feedback: {feedback_7.content}; correct: {correct_7.content}")
        if correct_7.content.strip() == "True":
            break
        cot_inputs_7.extend([thinking_7, answer_7, feedback_7])
        thinking_7, answer_7 = await reflexion_cot_agent(cot_inputs_7, instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflexion_cot_agent.id}, refining solution, thinking: {thinking_7.content}; answer: {answer_7.content}")

    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_7.content}; answer - {answer_7.content}")
    subtask_desc_7['response'] = {"thinking": thinking_7, "answer": answer_7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_7, answer_7, sub_tasks, agents)
    return final_answer, logs
