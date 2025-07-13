async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and formalize the geometric setup: Identify the angle theta at vertex B formed by sides AB and BC of triangle ABC. "
        "Model the chain of n tangent circles of equal radius r arranged sequentially inside this angle such that the first circle is tangent to side AB and the last circle is tangent to side BC. "
        "Define variables for the angle theta, the radius r, and the number of circles n. Carefully establish the geometric relationship between the radius, number of circles, and the angle theta, assuming the circles are tangent to both sides and to each other in sequence. "
        "Explicitly represent the positions of the circle centers along the angle bisector and the spacing between them. Avoid assuming any additional properties of the triangle beyond what is implied by the circle arrangement."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing geometric setup, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Analyze the scaling relationship between the two given configurations: one with 8 circles of radius 34 and another with 2024 circles of radius 1 arranged similarly inside the angle at B. "
        "Use this to derive an equation relating the angle theta, the radius r, and the number of circles n. Confirm that the angle theta is fixed and independent of the radius and number of circles, and that the product n·r is constant. "
        "Avoid mixing assumptions about the triangle's inradius at this stage. Provide clear reasoning to support the constancy of theta and the proportionality between n and 1/r."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1.content, answer_0_1.content], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, analyzing scaling relationship, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent and correct solutions for scaling relationship." , is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing scaling relationship, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Derive a correct formula for the inradius r_in of triangle ABC in terms of the angle theta at vertex B and the radius r of the tangent circles. "
        "Explicitly use the key geometric relationship that the distance from vertex B to the incircle center along the angle bisector is BI = r_in / sin(theta/2). "
        "Model the distance from B to the center of the last circle in the chain as d_n = r / sin(theta/2) + 2r(n - 1). "
        "Set these distances equal to relate r_in, r, n, and theta, leading to the formula r_in = r + 2r(n - 1) sin(theta/2). "
        "Include detailed derivation steps and avoid incorrect assumptions equating d_n directly to r_in without the sin(theta/2) factor. "
        "Immediately after derivation, perform explicit sanity checks: verify that for n=1, r_in = r, and check dimensional consistency and behavior for large n. "
        "Document these validation steps clearly to ensure geometric and algebraic correctness before proceeding."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2.content, answer_0_2.content], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, deriving inradius formula, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    reflect_instruction_1_2 = (
        "Sub-task 2: Conduct a dedicated verification and reflection on the derived inradius formula from subtask_1. "
        "Cross-check the formula against known geometric properties of incircles and tangent circles in an angle. "
        "Test limiting cases and confirm consistency with the problem's given data (8 circles of radius 34 and 2024 circles of radius 1). "
        "Identify and correct any inconsistencies or errors found. This subtask acts as a critical review step to prevent propagation of errors into algebraic manipulation. "
        "Provide a clear statement of formula correctness or necessary corrections."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round
    cot_inputs_1_2 = [taskInfo, thinking_1_1.content, answer_1_1.content]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": reflect_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, reflect_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, verifying inradius formula, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    critic_inst_1_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_1_2):
        feedback_1_2, correct_1_2 = await critic_agent_1_2([taskInfo, thinking_1_2.content, answer_1_2.content], "Please review and provide the limitations of provided solutions" + critic_inst_1_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, providing feedback, thinking: {feedback_1_2.content}; answer: {correct_1_2.content}")
        if correct_1_2.content == "True":
            break
        cot_inputs_1_2.extend([thinking_1_2.content, answer_1_2.content, feedback_1_2.content])
        thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining inradius formula, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_instruction_1_3 = (
        "Sub-task 3: Express the inradius r_in as a reduced fraction m/n by substituting the known values (n=8, r=34 or n=2024, r=1) into the validated formula. "
        "Simplify the fraction to lowest terms, ensuring m and n are relatively prime positive integers. Avoid computational errors or premature rounding. "
        "Provide detailed algebraic steps and justification for simplification. Ensure the fraction form is consistent with the geometric interpretation and previous validation."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_2.content, answer_1_2.content], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, simplifying fraction for inradius, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Compute the sum m + n from the simplified fraction representing the inradius. "
        "Verify the correctness of the simplification and the final sum by cross-referencing with previous results and sanity checks. "
        "Provide the final answer clearly and unambiguously. Include a brief summary of the entire solution process and validation results to confirm the reliability of the answer."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_3.content, answer_1_3.content], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, computing sum m+n, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize and finalize the sum m+n." , is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing sum m+n, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    debate_instr_3_1 = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_instruction_3_1 = "Sub-task 1: Finalize the answer for the sum m+n." + debate_instr_3_1
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_3_1, answer_3_1 = await agent([taskInfo, thinking_2_1.content, answer_2_1.content], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking_2_1.content, answer_2_1.content] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_3_1, answer_3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, finalizing answer, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
            all_thinking_3_1[r].append(thinking_3_1)
            all_answer_3_1[r].append(answer_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 1: Provide the final answer for the sum m+n." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, computing final answer, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
