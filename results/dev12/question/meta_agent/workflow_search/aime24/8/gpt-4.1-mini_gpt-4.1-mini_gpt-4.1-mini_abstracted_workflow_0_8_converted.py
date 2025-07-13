async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent_stage0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_instruction_0_1 = "Sub-task 1: Formally represent the game setup: define the initial state as a stack of n tokens (1 ≤ n ≤ 2024), the players Alice and Bob alternating turns with Alice starting first, and the allowed moves (removing either 1 or 4 tokens). Clearly state the winning condition as the player who removes the last token wins."
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_stage0([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0.id}, formalizing game setup, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = "Sub-task 2: Define the concepts of winning and losing positions in the context of this game: a position is losing if the player to move cannot force a win, and winning otherwise. Emphasize that Bob has a guaranteed winning strategy if the initial position is losing for Alice (the first player)."
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_stage0([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0.id}, defining winning/losing positions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_0_3 = "Sub-task 3: Identify and formalize the state transitions based on allowed moves: from a position with k tokens, the next positions are k-1 or k-4 tokens (if valid). Highlight that these transitions define the structure of the game’s state graph and govern the classification of positions."
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_stage0([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0.id}, formalizing state transitions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    N_sc = self.max_sc
    cot_agents_stage1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    cot_sc_instruction_1_1 = "Sub-task 1: Using the formalization from stage_0, compute the classification (winning or losing) of positions for all n from 0 up to 2024 by applying backward induction or dynamic programming, starting from the base case (0 tokens is losing since no moves are possible)."
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2, answer_0_2, thinking_0_3, answer_0_3],
        "agent_collaboration": "CoT | SC_CoT"
    }
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_stage1_1[i]([taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_1[i].id}, computing position classification, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent and correct classification of positions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_agents_stage1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    cot_sc_instruction_1_2 = "Sub-task 2: Analyze the pattern of winning and losing positions obtained in subtask_1 to identify any periodicity or modular arithmetic pattern that can simplify the classification for large n."
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "CoT | SC_CoT | Reflexion"
    }
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_stage1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_2[i].id}, analyzing pattern of positions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent pattern analysis.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_agent_stage1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    reflect_inst_1_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_3 = "Sub-task 3: Formulate a closed-form or modular condition that characterizes losing positions (positions where the player to move loses) based on the pattern analysis, enabling efficient counting without enumerating all positions explicitly." + reflect_inst_1_3
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_stage1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage1_3.id}, formulating closed-form condition, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback])
        thinking_1_3, answer_1_3 = await cot_agent_stage1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage1_3.id}, refining closed-form condition, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_agent_stage2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_2_1 = "Sub-task 1: Apply the characterization from stage_1.subtask_3 to count the number of integers n ≤ 2024 for which the initial position is losing for Alice, i.e., Bob has a winning strategy. Provide the final count as the answer to the query."
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3, answer_1_3],
        "agent_collaboration": "CoT | Reflexion"
    }
    reflect_inst_2_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = cot_instruction_2_1 + reflect_inst_2_1
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_1 = [taskInfo, thinking_1_3, answer_1_3]
    thinking_2_1, answer_2_1 = await cot_agent_stage2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2_1.id}, counting losing positions, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback])
        thinking_2_1, answer_2_1 = await cot_agent_stage2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2_1.id}, refining count of losing positions, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
