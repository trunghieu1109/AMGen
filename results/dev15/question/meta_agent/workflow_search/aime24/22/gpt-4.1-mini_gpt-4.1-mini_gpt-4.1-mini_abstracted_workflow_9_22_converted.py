async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    N_sc = self.max_sc

    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_debate = self.max_round

    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    # Stage 0 - Subtask 1: Formal representation and constraints (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent the list and its elements, defining variables for the list length, element values, and their frequencies. "
        "Extract and clearly state the constraints given: sum of elements equals 30, unique mode is 9, median is a positive integer not in the list, and all elements are positive integers."
    )
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing list and constraints, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0_subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Stage 0 - Subtask 2: Analyze median implications (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Analyze the implications of the median being a positive integer not in the list, "
        "deducing that the list length must be even and that the median equals the average of the two middle elements, "
        "which are distinct and do not equal the median itself."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, analyzing median implications, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task stage_0_subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 0 - Subtask 3: Express unique mode condition (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Formally express the unique mode condition, ensuring that the frequency of 9 is strictly greater than the frequency of any other element in the list, "
        "and consider how this restricts the possible frequency distributions of elements."
    )
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, expressing unique mode condition, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task stage_0_subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1 - Subtask 1: Determine possible even lengths (Reflexion)
    reflect_inst_1_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_1 = (
        "Sub-task 1: Determine possible even lengths of the list that allow a median not in the list and satisfy the sum constraint of 30, "
        "considering the minimum and maximum sums achievable for different lengths." + reflect_inst_1_1
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_1_1 = [taskInfo, thinking_0_2]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, determining possible even lengths, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(self.max_round):
        feedback_1_1, correct_1_1 = await critic_agent_1_1([taskInfo, thinking_1_1],
                                                        "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback_1_1.content}; answer: {correct_1_1.content}")
        if correct_1_1.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, feedback_1_1])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining possible even lengths, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1_subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Stage 1 - Subtask 2: Identify candidate middle elements (Reflexion)
    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Identify candidate values for the two middle elements whose average is the median (a positive integer not in the list), "
        "ensuring these two elements are distinct positive integers and do not equal the median." + reflect_inst_1_2
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_1_2 = [taskInfo, thinking_0_2]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, identifying candidate middle elements, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    for i in range(self.max_round):
        feedback_1_2, correct_1_2 = await critic_agent_1_1([taskInfo, thinking_1_2],
                                                        "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback_1_2.content}; answer: {correct_1_2.content}")
        if correct_1_2.content == "True":
            break
        cot_inputs_1_2.extend([thinking_1_2, feedback_1_2])
        thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining candidate middle elements, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1_subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 1 - Subtask 3: Construct possible lists (SC-CoT)
    cot_sc_instruction_1_3 = (
        "Sub-task 3: Construct possible lists including multiple 9s (to satisfy unique mode) and other positive integers, "
        "ensuring the sum is 30 and the median condition is met, and verify the uniqueness of the mode 9 in each candidate list."
    )
    subtask_desc_1_3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_1.content, thinking_1_2.content, thinking_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_1_1, thinking_1_2, thinking_0_3], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, constructing candidate lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and choose the most consistent candidate lists." + cot_sc_instruction_1_3, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_3.id}, synthesizing candidate lists, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task stage_1_subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2 - Subtask 1: Simplify and verify candidate lists (Debate)
    debate_instr_2_1 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2_1 = "Sub-task 1: Simplify the candidate lists by verifying the sum of elements equals 30, the median is the average of the two middle elements and is not in the list, and the mode is uniquely 9, discarding any lists that do not satisfy all conditions." + debate_instr_2_1
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "Debate"
    }
    all_thinking_2_1 = [[] for _ in range(N_debate)]
    all_answer_2_1 = [[] for _ in range(N_debate)]
    for r in range(N_debate):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_3], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3] + all_thinking_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying candidate lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 1: Final verification and simplification of candidate lists." + debate_instr_2_1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_1.id}, final verification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task stage_2_subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Stage 2 - Subtask 2: Compute sum of squares (Reflexion)
    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: From the verified candidate list(s), compute the sum of the squares of all the items in the list." + reflect_inst_2_2
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, computing sum of squares, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(self.max_round):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2],
                                                        "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining sum of squares, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task stage_2_subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Stage 3 - Subtask 1: Aggregate and present final answer (SC-CoT)
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Aggregate the computed sum of squares from the verified list(s) and present the final answer clearly, "
        "ensuring all problem conditions are met and the solution is consistent."
    )
    subtask_desc_3_1 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_2_2, answer_2_2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, aggregating final answer, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_thinkings_3_1, "Sub-task 1: Final aggregation and presentation of the answer." + cot_sc_instruction_3_1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_3_1.id}, final aggregation, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3_subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
