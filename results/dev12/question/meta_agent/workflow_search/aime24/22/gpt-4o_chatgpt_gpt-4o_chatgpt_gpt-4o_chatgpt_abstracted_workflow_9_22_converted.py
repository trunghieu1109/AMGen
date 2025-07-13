async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Constraint Analysis
    cot_instruction = "Sub-task 1: Identify and clearly state the constraints: the list consists of positive integers, the sum is 30, the unique mode is 9, and the median is a positive integer not in the list."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: List Configuration Exploration
    # Sub-task 2.1: Identify possible even list lengths starting from the smallest viable (n=4)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2.1: Identify possible even list lengths starting from the smallest viable (n=4) and justify their validity based on the constraints." + reflect_inst
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1]
    subtask_desc2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2_1, answer2_1 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, identifying list lengths, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2_1, answer2_1], 
                                       "Please review and provide the limitations of provided solutions" + critic_inst, 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2_1, answer2_1, feedback])
        thinking2_1, answer2_1 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining list lengths, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {
        "thinking": thinking2_1,
        "answer": answer2_1
    }
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Sub-task 2.2: Generate candidate lists for each even length n
    cot_sc_instruction = "Sub-task 2.2: For each candidate even length n, generate multiple candidate lists satisfying the sum and unique mode constraints."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 2_1", "answer of subtask 2_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2_2, answer2_2 = await cot_agents[i]([taskInfo, thinking2_1, answer2_1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, generating candidate lists, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
        possible_answers.append(answer2_2)
        possible_thinkings.append(thinking2_2)
    final_instr = "Given all the above thinking and answers, find the most consistent and correct solutions for the candidate lists"
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2_2, answer2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers + possible_thinkings, 
                                                 "Sub-task 2.2: Synthesize and choose the most consistent candidate lists" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {
        "thinking": thinking2_2,
        "answer": answer2_2
    }
    logs.append(subtask_desc2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Sub-task 2.3: Verify the median condition for each candidate list
    cot_sc_instruction = "Sub-task 2.3: For each candidate list, rigorously verify the median condition: the median is a positive integer not in the list."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2_3 = {
        "subtask_id": "subtask_2_3",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 2_2", "answer of subtask 2_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2_3, answer2_3 = await cot_agents[i]([taskInfo, thinking2_2, answer2_2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, verifying median condition, thinking: {thinking2_3.content}; answer: {answer2_3.content}")
        possible_answers.append(answer2_3)
        possible_thinkings.append(thinking2_3)
    final_instr = "Given all the above thinking and answers, find the most consistent and correct solutions for the median condition"
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2_3, answer2_3 = await final_decision_agent_2_3([taskInfo] + possible_answers + possible_thinkings, 
                                                 "Sub-task 2.3: Synthesize and choose the most consistent median condition verification" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking2_3.content}; answer - {answer2_3.content}")
    subtask_desc2_3['response'] = {
        "thinking": thinking2_3,
        "answer": answer2_3
    }
    logs.append(subtask_desc2_3)
    print("Step 2.3: ", sub_tasks[-1])

    # Sub-task 2.4: Select the valid configuration with the smallest n
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2_4 = "Sub-task 2.4: Select the valid configuration with the smallest n that meets all constraints." + debate_instr
    debate_agents_2_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_2_4 = self.max_round
    all_thinking2_4 = [[] for _ in range(N_max_2_4)]
    all_answer2_4 = [[] for _ in range(N_max_2_4)]
    subtask_desc2_4 = {
        "subtask_id": "subtask_2_4",
        "instruction": debate_instruction_2_4,
        "context": ["user query", "thinking of subtask 2_3", "answer of subtask 2_3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_4):
        for i, agent in enumerate(debate_agents_2_4):
            if r == 0:
                thinking2_4, answer2_4 = await agent([taskInfo, thinking2_3, answer2_3], 
                                           debate_instruction_2_4, r, is_sub_task=True)
            else:
                input_infos_2_4 = [taskInfo, thinking2_3, answer2_3] + all_thinking2_4[r-1] + all_answer2_4[r-1]
                thinking2_4, answer2_4 = await agent(input_infos_2_4, debate_instruction_2_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting valid configuration, thinking: {thinking2_4.content}; answer: {answer2_4.content}")
            all_thinking2_4[r].append(thinking2_4)
            all_answer2_4[r].append(answer2_4)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2_4, answer2_4 = await final_decision_agent_2_4([taskInfo] + all_thinking2_4[-1] + all_answer2_4[-1], 
                                                 "Sub-task 2.4: Select the valid configuration" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.4 output: thinking - {thinking2_4.content}; answer - {answer2_4.content}")
    subtask_desc2_4['response'] = {
        "thinking": thinking2_4,
        "answer": answer2_4
    }
    logs.append(subtask_desc2_4)
    print("Step 2.4: ", sub_tasks[-1])

    # Stage 3: Sum of Squares Calculation
    cot_sc_instruction = "Sub-task 3.1: Calculate the sum of the squares of the integers in the selected list configuration."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc3_1 = {
        "subtask_id": "subtask_3_1",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 2_4", "answer of subtask 2_4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3_1, answer3_1 = await cot_agents[i]([taskInfo, thinking2_4, answer2_4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, calculating sum of squares, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        possible_answers.append(answer3_1)
        possible_thinkings.append(thinking3_1)
    final_instr = "Given all the above thinking and answers, find the most consistent and correct solutions for the sum of squares"
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo] + possible_answers + possible_thinkings, 
                                                 "Sub-task 3.1: Synthesize and choose the most consistent sum of squares calculation" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {
        "thinking": thinking3_1,
        "answer": answer3_1
    }
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_1, answer3_1, sub_tasks, agents)
    return final_answer, logs