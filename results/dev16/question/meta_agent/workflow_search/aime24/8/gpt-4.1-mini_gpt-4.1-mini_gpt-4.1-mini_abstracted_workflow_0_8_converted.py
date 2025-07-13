async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    # Stage 0: Formalization and conceptual definitions

    # Subtask 1: Formalize the game setup (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the game setup: specify the initial state as a stack of n tokens (1 ≤ n ≤ 2024), "
        "players Alice and Bob alternating turns with Alice starting, allowed moves (removing 1 or 4 tokens), and the winning condition (last token removed wins). "
        "Avoid solving or analyzing the game at this stage; focus on formalizing problem elements and constraints.")
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formalizing game setup, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Explain winning and losing positions concept (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Explain the concepts of winning and losing positions in impartial combinatorial games under normal play convention. "
        "Define losing position as one where the player to move cannot force a win, winning otherwise. Emphasize Bob can guarantee a win if the initial position is losing for Alice. "
        "Avoid introducing specific positions or computations; focus on conceptual clarity.")
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, explaining winning/losing positions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Recursive characterization of losing positions (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Identify the relationship between allowed moves {1,4} and transitions between game states. "
        "Formulate recursive characterization: a position is losing if all reachable positions by allowed moves are winning, and winning if at least one reachable position is losing. "
        "Avoid computing explicit positions; focus on formalizing recursive structure and constraints.")
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent([taskInfo, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating recursive characterization, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Compute base cases and identify pattern

    # Subtask 1: Compute base cases n=0..4 (CoT)
    cot_instruction_1_1 = (
        "Sub-task 1: Compute base cases for game states n=0,1,2,3,4 using recursive definition. "
        "Determine winning or losing status for these small values. Avoid jumping to conclusions; just compute and record.")
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing base cases, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 2: Compute status for n=1..30 iteratively (CoT with Self-Consistency)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Using base cases, iteratively compute winning/losing status for n=1 to 30. "
        "Document results carefully and look for periodicity or repetition. Avoid assumptions without evidence.")
    N_sc = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, computing n=1..30, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent answer for iterative computation of n=1..30.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing iterative computation, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 3: Analyze computed sequence to identify pattern (Reflexion)
    reflect_inst_1_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Analyze the computed sequence of winning and losing positions to identify any periodicity or closed-form characterization. "
        "Formulate a hypothesis about the pattern (e.g., losing positions occur at n congruent to a residue modulo a fixed number). Avoid finalizing without verification. "
        + reflect_inst_1_3)
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, hypothesizing pattern, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3], "Please review and provide limitations of the solution. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback_1_3.content}; correct: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining pattern, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 4: Verify hypothesized pattern beyond initial range (Reflexion)
    reflect_inst_1_4 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_4 = (
        "Sub-task 4: Verify the hypothesized pattern of losing positions by checking additional values beyond initial computed range to confirm consistency. "
        "Avoid relying on partial data; ensure pattern holds for a large sample. "
        + reflect_inst_1_4)
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_4 = self.max_round
    cot_inputs_1_4 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1_subtask_4",
        "instruction": cot_reflect_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_1_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, verifying pattern, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    for i in range(N_max_1_4):
        feedback_1_4, correct_1_4 = await critic_agent_1_4([taskInfo, thinking_1_4], "Please review and provide limitations of the verification. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_4.id}, feedback: {feedback_1_4.content}; correct: {correct_1_4.content}")
        if correct_1_4.content == "True":
            break
        cot_inputs_1_4.extend([thinking_1_4, feedback_1_4])
        thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_1_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, refining verification, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 7: ", sub_tasks[-1])

    # Stage 2: Closed-form characterization and counting

    # Subtask 1: Formulate closed-form modular condition (CoT)
    cot_instruction_2_1 = (
        "Sub-task 1: Formulate a closed-form or modular arithmetic condition characterizing all losing positions for the game with moves {1,4}. "
        "Clearly state this condition as a function of n. Avoid ambiguity or partial descriptions; condition should be precise and verifiable.")
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent([taskInfo, thinking_1_4, answer_1_4], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating closed-form condition, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 8: ", sub_tasks[-1])

    # Subtask 2: Compute count of n ≤ 2024 satisfying losing condition (CoT)
    cot_instruction_2_2 = (
        "Sub-task 2: Using the characterization of losing positions, compute the number of positive integers n ≤ 2024 that satisfy the losing position condition. "
        "This count corresponds to the number of initial positions where Bob can guarantee a win. Avoid manual counting; use arithmetic or algorithmic methods.")
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing count of losing positions, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 9: ", sub_tasks[-1])

    # Subtask 3: Summarize final result (CoT)
    cot_instruction_2_3 = (
        "Sub-task 3: Summarize the final result: state the count of n ≤ 2024 for which Bob has a winning strategy, "
        "and briefly explain how the characterization and counting were performed. Avoid introducing new analysis; focus on clear presentation of conclusion.")
    subtask_desc_2_3 = {
        "subtask_id": "stage_2_subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_3, answer_2_3 = await cot_agent([taskInfo, thinking_2_2, answer_2_2], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, summarizing final result, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
