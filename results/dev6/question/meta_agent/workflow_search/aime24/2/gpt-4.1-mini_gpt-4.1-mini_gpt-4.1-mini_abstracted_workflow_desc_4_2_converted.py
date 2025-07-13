async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Derive a formal representation of the problem. "
        "Label the octagon vertices 0 to 7, define blue vertices set B and red vertices set R as complement. "
        "Represent rotations as elements of cyclic group of order 8 acting on vertex indices. "
        "Formally express the condition that there exists a rotation r such that r(B) is a subset of R. "
        "Clarify assumptions including identity rotation and independent equal probability coloring."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_0_1}")
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving formal representation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Validate the formal representation from Sub-task 1 by checking consistency with the problem statement. "
        "Ensure rotation action and coloring definitions correctly capture the problem constraints. "
        "Confirm the problem reduces to checking existence of a rotation mapping blue vertices to originally red vertices."
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
    print(f"Logging before SC-CoT agents call: {subtask_desc_0_2}")
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, validating formal representation, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent answer for formal representation validation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Identify and enumerate all rotations of the octagon (including identity) and their cycle structures. "
        "For each rotation, characterize vertex orbits and how these orbits affect coloring constraints. "
        "Determine cycle decomposition of each rotation and understand how rotations permute vertex indices."
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
    print(f"Logging before SC-CoT agents call: {subtask_desc_1_1}")
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating rotations and cycle structures, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent answer for rotation enumeration.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_instruction_1_2a = (
        "Sub-task 2a: For each rotation, determine the number of colorings fixed by that rotation under the condition that blue vertices, when rotated, lie entirely on red vertices. "
        "Analyze fixed points and constraints imposed by each rotation's cycle structure. "
        "Avoid direct summation; prepare for Burnside's lemma application."
    )
    cot_agent_1_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2a = {
        "subtask_id": "stage_1.subtask_2a",
        "instruction": cot_instruction_1_2a,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_1_2a}")
    thinking_1_2a, answer_1_2a = await cot_agent_1_2a([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2a.id}, counting colorings fixed by each rotation, thinking: {thinking_1_2a.content}; answer: {answer_1_2a.content}")
    sub_tasks.append(f"Sub-task 1.2a output: thinking - {thinking_1_2a.content}; answer - {answer_1_2a.content}")
    subtask_desc_1_2a['response'] = {"thinking": thinking_1_2a, "answer": answer_1_2a}
    logs.append(subtask_desc_1_2a)
    print("Step 1.2a: ", sub_tasks[-1])

    debate_instruction_1_2b = (
        "Sub-task 2b: Apply Burnside's lemma to compute the exact number of colorings for which there exists at least one rotation satisfying the condition. "
        "Average the counts of colorings fixed by each rotation to correctly account for overlaps and avoid overcounting. "
        "Explicitly document the application and resulting count. Avoid using any asserted constants; derive all counts rigorously."
    )
    debate_agents_1_2b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_2b = self.max_round
    all_thinking_1_2b = [[] for _ in range(N_max_1_2b)]
    all_answer_1_2b = [[] for _ in range(N_max_1_2b)]
    subtask_desc_1_2b = {
        "subtask_id": "stage_1.subtask_2b",
        "instruction": debate_instruction_1_2b,
        "context": ["user query", thinking_1_2a.content, answer_1_2a.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before Debate agents call: {subtask_desc_1_2b}")
    for r in range(N_max_1_2b):
        for i, agent in enumerate(debate_agents_1_2b):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2a, answer_1_2a], debate_instruction_1_2b, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2a, answer_1_2a] + all_thinking_1_2b[r-1] + all_answer_1_2b[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_1_2b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, applying Burnside's lemma, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_1_2b[r].append(thinking_i)
            all_answer_1_2b[r].append(answer_i)
    final_decision_agent_1_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2b, answer_1_2b = await final_decision_agent_1_2b([taskInfo] + all_thinking_1_2b[-1] + all_answer_1_2b[-1], "Sub-task 2b: Final synthesis applying Burnside's lemma.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2b output: thinking - {thinking_1_2b.content}; answer - {answer_1_2b.content}")
    subtask_desc_1_2b['response'] = {"thinking": thinking_1_2b, "answer": answer_1_2b}
    logs.append(subtask_desc_1_2b)
    print("Step 1.2b: ", sub_tasks[-1])

    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Verify the computed count of favorable colorings by cross-checking with an independent method. "
        "Perform brute-force enumeration of all 256 colorings or compare with known combinatorial identities. "
        "Confirm the count matches the Burnside's lemma result. If discrepancies arise, revisit previous subtasks."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2b, answer_1_2b]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2b.content, answer_1_2b.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before Reflexion CoT agent call: {subtask_desc_1_3}")
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, verifying count by brute-force, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback, correct = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review and provide limitations of provided solutions. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining verification, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_sc_instruction_1_4 = (
        "Sub-task 4: Calculate the probability by dividing the verified number of favorable colorings by total colorings (256). "
        "Express this probability as a reduced fraction m/n, ensuring m and n are relatively prime. "
        "Document the simplification process clearly."
    )
    N_sc_1_4 = self.max_sc
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_4)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agents call: {subtask_desc_1_4}")
    for i in range(N_sc_1_4):
        thinking_i, answer_i = await cot_agents_1_4[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, calculating probability fraction, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_4.append(answer_i)
        possible_thinkings_1_4.append(thinking_i)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_answers_1_4 + possible_thinkings_1_4, "Sub-task 4: Synthesize and choose the most consistent answer for probability fraction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    debate_instruction_2_1 = (
        "Sub-task 1: Simplify the fraction m/n obtained from Stage 1 Sub-task 4 to lowest terms by finding GCD. "
        "Verify m and n are relatively prime with explicit justification."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before Debate agents call: {subtask_desc_2_1}")
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_4, answer_1_4], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_4, answer_1_4] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying fraction, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Final decision on simplified fraction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Compute the sum m + n as required by the problem statement, based on simplified fraction from Sub-task 1. "
        "Use reflexion to ensure correctness and improve from previous feedback."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before Reflexion CoT agent call: {subtask_desc_2_2}")
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, computing sum m+n, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide limitations of provided solutions. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining sum m+n, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate the final result m + n and verify correctness of all previous steps. "
        "Cross-check reasoning and calculations to ensure no errors. Provide final answer with brief justification referencing verification and Burnside's lemma application."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_1_4.content, answer_1_4.content, thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_3_1}")
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_0_2, answer_0_2, thinking_1_4, answer_1_4, thinking_2_2, answer_2_2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, aggregating final result, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    N_sc_3_1 = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_1)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1_sc = {
        "subtask_id": "stage_3.subtask_1_sc",
        "instruction": "Sub-task 1 (SC): Verify final answer consistency and correctness by multiple independent reasoning attempts.",
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before SC-CoT agents call: {subtask_desc_3_1_sc}")
    for i in range(N_sc_3_1):
        thinking_i, answer_i = await cot_agents_3_1[i]([taskInfo, thinking_3_1, answer_3_1], subtask_desc_3_1_sc['instruction'], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, verifying final answer, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1_sc, answer_3_1_sc = await final_decision_agent_3_1([taskInfo] + possible_answers_3_1 + possible_thinkings_3_1, "Sub-task 1 (SC): Final synthesis and verification of answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 SC output: thinking - {thinking_3_1_sc.content}; answer - {answer_3_1_sc.content}")
    subtask_desc_3_1_sc['response'] = {"thinking": thinking_3_1_sc, "answer": answer_3_1_sc}
    logs.append(subtask_desc_3_1_sc)
    print("Step 3.1 SC: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1_sc, answer_3_1_sc, sub_tasks, agents)
    return final_answer, logs
