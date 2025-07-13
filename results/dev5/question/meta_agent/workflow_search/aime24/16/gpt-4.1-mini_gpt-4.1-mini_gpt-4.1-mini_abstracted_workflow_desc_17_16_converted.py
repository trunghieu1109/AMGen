async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Precisely analyze the geometric configuration of points O (circumcenter), I (incenter), and A (vertex) in triangle ABC. "
        "Use vector or coordinate geometry to express the positions of O and I relative to A, incorporating the given perpendicularity condition IA perpendicular to OI. "
        "Derive an explicit expression for the length OI using Euler's formula (OI^2 = R(R - 2r)) and confirm its numeric value. "
        "Avoid any assumptions about triangle shape or angle measures at this stage. Document all intermediate steps and assumptions clearly to enable later verification."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_1_1}")
    thinking1_1, answer1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, analyzing geometric configuration, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Derive the length IA using the relation IA^2 = R^2 - OI^2, where R is the circumradius and OI is from subtask 1.1. "
        "Then, using the inradius r and the formula IA = r / sin(A/2), compute sin(A/2) precisely. "
        "From sin(A/2), derive exact expressions for cos A and sin A using half-angle formulas. "
        "Avoid numeric substitution until all symbolic relations are confirmed consistent. Carefully check for algebraic consistency and document all derivations."
    )
    N_sc = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking1_1.content, answer1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agents call: {subtask_desc_1_2}")
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(N_sc):
        thinking1_2, answer1_2 = await cot_agents_1_2[i]([taskInfo, thinking1_1, answer1_1], cot_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, deriving IA and sin(A/2), thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2)
        possible_thinkings_1_2.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_2 = "Given all the above thinking and answers, find the most consistent and correct derivations for IA, sin(A/2), cos A, and sin A."
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 1.2: Synthesize and choose the most consistent answer for IA and angle A." + final_instr_1_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_instruction_1_3 = (
        "Sub-task 3: Express sin B and sin C in terms of sin(A/2), cos A, and other known parameters using angle sum properties and triangle relations. "
        "Derive an exact symbolic expression for sin B sin C without premature numeric substitution. Prepare these expressions for cross-validation. "
        "Avoid assuming any numeric values for angles B or C before verification."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking1_2.content, answer1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agents call: {subtask_desc_1_3}")
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    for i in range(N_sc):
        thinking1_3, answer1_3 = await cot_agents_1_3[i]([taskInfo, thinking1_2, answer1_2], cot_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, expressing sin B sin C, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
        possible_answers_1_3.append(answer1_3)
        possible_thinkings_1_3.append(thinking1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_3 = "Given all the above thinking and answers, find the most consistent and correct symbolic expression for sin B sin C."
    thinking1_3, answer1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 1.3: Synthesize and choose the most consistent answer for sin B sin C." + final_instr_1_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking1_3, "answer": answer1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    debate_instruction_1_4 = (
        "Sub-task 4: Conduct a multi-agent Debate to cross-validate the derived value of sin B sin C from subtask 1.3. "
        "Agents should independently verify the expression using alternative geometric relations such as area formulas, half-angle identities, or coordinate geometry. "
        "The Debate should aim to reach consensus on the correctness of sin B sin C before proceeding. "
        "If inconsistencies arise, agents must collaboratively identify and correct errors in earlier derivations."
    )
    debate_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_4 = self.max_round
    all_thinking_1_4 = [[] for _ in range(N_max_1_4)]
    all_answer_1_4 = [[] for _ in range(N_max_1_4)]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_instruction_1_4,
        "context": ["user query", thinking1_3.content, answer1_3.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before Debate agents call: {subtask_desc_1_4}")
    for r in range(N_max_1_4):
        for i, agent in enumerate(debate_agents_1_4):
            if r == 0:
                thinking_1_4, answer_1_4 = await agent([taskInfo, thinking1_3, answer1_3], debate_instruction_1_4, r, is_sub_task=True)
            else:
                input_infos_1_4 = [taskInfo, thinking1_3, answer1_3] + all_thinking_1_4[r-1] + all_answer_1_4[r-1]
                thinking_1_4, answer_1_4 = await agent(input_infos_1_4, debate_instruction_1_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating sin B sin C, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
            all_thinking_1_4[r].append(thinking_1_4)
            all_answer_1_4[r].append(answer_1_4)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_4 = "Given all the above debate thinking and answers, reason carefully and provide a final consensus on sin B sin C."
    thinking1_4, answer1_4 = await final_decision_agent_1_4([taskInfo] + all_thinking_1_4[-1] + all_answer_1_4[-1], "Sub-task 1.4: Final consensus on sin B sin C." + final_instr_1_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking1_4.content}; answer - {answer1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking1_4, "answer": answer1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    cot_instruction_1_5 = (
        "Sub-task 5: Using the confirmed values from the Debate, compute side BC = a = 2R sin A. "
        "Then, apply the Law of Cosines and area relations to derive an exact expression for the product AB · AC = bc. "
        "Avoid direct numeric substitution until all symbolic relations are verified. Document all intermediate steps and assumptions clearly."
    )
    cot_agents_1_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_5 = {
        "subtask_id": "stage_1.subtask_5",
        "instruction": cot_instruction_1_5,
        "context": ["user query", thinking1_4.content, answer1_4.content, thinking1_2.content, answer1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agents call: {subtask_desc_1_5}")
    possible_answers_1_5 = []
    possible_thinkings_1_5 = []
    for i in range(N_sc):
        thinking1_5, answer1_5 = await cot_agents_1_5[i]([taskInfo, thinking1_4, answer1_4, thinking1_2, answer1_2], cot_instruction_1_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_5[i].id}, computing AB*AC, thinking: {thinking1_5.content}; answer: {answer1_5.content}")
        possible_answers_1_5.append(answer1_5)
        possible_thinkings_1_5.append(thinking1_5)
    final_decision_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_5 = "Given all the above thinking and answers, find the most consistent and correct expression for AB times AC."
    thinking1_5, answer1_5 = await final_decision_agent_1_5([taskInfo] + possible_answers_1_5 + possible_thinkings_1_5, "Sub-task 1.5: Synthesize and choose the most consistent answer for AB*AC." + final_instr_1_5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.5 output: thinking - {thinking1_5.content}; answer - {answer1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking1_5, "answer": answer1_5}
    logs.append(subtask_desc_1_5)
    print("Step 1.5: ", sub_tasks[-1])

    reflexion_instruction_2_1 = (
        "Sub-task 6: Substitute the known numeric values of R = 13 and r = 6 into the expressions derived in stage 1 to compute the numeric value of AB · AC. "
        "Perform algebraic simplifications carefully, ensuring all geometric constraints and inequalities (e.g., triangle inequalities, positivity of sides) are satisfied. "
        "Explicitly verify that the perpendicularity condition IA perpendicular to OI holds numerically using coordinate or vector methods by reconstructing points A, I, and O. Document the verification process in detail."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflexion_instruction_2_1,
        "context": ["user query", thinking1_5.content, answer1_5.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before Reflexion CoT agent call: {subtask_desc_2_1}")
    thinking2_1, answer2_1 = await cot_agent_2_1([taskInfo, thinking1_5, answer1_5], reflexion_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, numeric computation and verification, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking2_1, answer2_1], "Please review and provide limitations of the provided solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content.strip() == "True":
            break
        thinking2_1, answer2_1 = await cot_agent_2_1([taskInfo, thinking1_5, answer1_5, thinking2_1, answer2_1, feedback], reflexion_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining numeric solution, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflexion_instruction_2_2 = (
        "Sub-task 7: Perform a comprehensive verification of the final answer AB · AC by cross-checking with alternative geometric relations such as the formula for the area of the triangle, the inradius formula (Area = r * semiperimeter), and Euler’s inequality. "
        "Confirm that the computed side lengths and angles satisfy all given conditions, including the perpendicularity IA perpendicular to OI. "
        "If any inconsistency is detected, initiate a feedback loop to revisit and correct earlier subtasks. Provide a final, justified statement of the answer along with the verification results."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflexion_instruction_2_2,
        "context": ["user query", thinking2_1.content, answer2_1.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before Reflexion CoT agent call: {subtask_desc_2_2}")
    thinking2_2, answer2_2 = await cot_agent_2_2([taskInfo, thinking2_1, answer2_1], reflexion_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying final answer, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking2_2, answer2_2], "Please review and provide limitations of the verification. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content.strip() == "True":
            break
        thinking2_2, answer2_2 = await cot_agent_2_2([taskInfo, thinking2_1, answer2_1, thinking2_2, answer2_2, feedback], reflexion_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining verification, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_2, answer2_2, sub_tasks, agents)
    return final_answer, logs
