async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Derive the parametric equations of the torus including an explicit vertical displacement parameter h along the axis of revolution relative to the sphere's center. "
        "Define h as the vertical shift of the center of the generating circle of radius 3 from the origin (center of the sphere). "
        "Express the torus parametrically in terms of angles u and v, incorporating h, and write down the implicit equation of the sphere centered at the origin with radius 11. "
        "Validate that these representations correctly model the torus and sphere geometry with the vertical shift h."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving parametric equations with vertical shift h, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Express the distance from the sphere's center (origin) to an arbitrary point on the vertically shifted torus as a function of parameters u, v, and h. "
        "Formulate the condition for external tangency between the torus and the sphere: the distance from the origin to points on the torus equals the sphere radius 11 along a circle parameterized by v (for fixed u). "
        "Derive the tangency condition equations involving h and v that must be satisfied simultaneously. Keep conditions general to capture all possible tangent configurations."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, formulating tangency conditions with h and v, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize tangency condition equations involving h and v.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Solve the system of equations derived from the tangency conditions to find the two distinct pairs (h, v) corresponding to the two external tangent configurations of the torus resting on the sphere. "
        "Verify that h values represent physically plausible vertical displacements and v lies within valid parameter ranges. Emphasize geometric interpretation as two distinct vertical positions producing external tangency."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, solving for (h,v) pairs, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Select consistent (h,v) solutions for tangent configurations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: For each tangent configuration (h, v) found, compute the radius of the tangent circle on the torus, denoted r_i and r_o respectively. "
        "Use the torus parameters and angle v to express the radius accurately. Verify these radii correspond to the geometric setup and that the difference r_i - r_o is meaningful."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, computing tangent circle radii r_i and r_o, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    reflexion_instruction_1_3 = (
        "Sub-task 3: Validate the computed tangent circle radii r_i and r_o by checking geometric feasibility and consistency with the problem statement. "
        "Confirm that the torus positions corresponding to h and the tangent circles correspond to external tangency without intersection. "
        "Verify the difference r_i - r_o is as expected and that tangent circles are distinct."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": reflexion_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, reflexion_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, validating tangent radii, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback, correct = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review the answer above and criticize on where might be wrong. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, reflexion_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining validation, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    debate_instruction_2_1 = (
        "Sub-task 1: Express the difference r_i - r_o as a fraction m/n in lowest terms, where m and n are positive integers with no common factors. "
        "Decompose the difference into numerator and denominator, simplify fully, and ensure positivity."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_3, answer_1_3], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3, answer_1_3] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying fraction, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Finalize simplified fraction m/n for r_i - r_o.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    reflexion_instruction_2_2 = (
        "Sub-task 2: Compute the sum m + n of the numerator and denominator from the simplified fraction representing r_i - r_o. "
        "Confirm the sum is correct and consistent with the fraction. Avoid arithmetic mistakes or misinterpretation."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflexion_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflexion_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, computing sum m+n, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review the answer above and criticize on where might be wrong. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflexion_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining sum m+n, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate all intermediate results and verify the final answer m + n. "
        "Provide a clear, concise final answer including the fraction form of r_i - r_o and the sum m + n. "
        "Cross-check all previous steps for consistency and correctness, especially the incorporation of the vertical displacement parameter h and the tangency conditions."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    final_inputs_3_1 = [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2, thinking_1_3, answer_1_3, thinking_2_1, answer_2_1, thinking_2_2, answer_2_2]
    thinking_3_1, answer_3_1 = await cot_agent_3_1(final_inputs_3_1, cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating final answer, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    for i, st in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", st)
    return final_answer, logs
