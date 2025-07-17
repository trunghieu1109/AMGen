async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 0_1: Extract and summarize the defining features of the original and variant decay processes, including particle types, decay products, and known spectral properties." + \
                    " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    all_thinking_0 = []
    all_answer_0 = []
    subtask_desc_0 = {
        "subtask_id": "subtask_0_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0):
        thinking0, answer0 = await agent([taskInfo], debate_instr_0, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, extracting and summarizing decay features, thinking: {thinking0.content}; answer: {answer0.content}")
        all_thinking_0.append(thinking0)
        all_answer_0.append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_0 + all_answer_0, 
                                                    "Sub-task 0_1: Extract and summarize decay features." + 
                                                    " Given all the above thinking and answers, reason over them carefully and provide a final answer.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_1 = "Sub-task 1_1: Analyze the physical and kinematic implications of replacing two lighter particles (2V) with a single massless particle (M) on energy and momentum conservation and phase space." + \
                    " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    all_thinking_1 = []
    all_answer_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": debate_instr_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1):
        thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instr_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, analyzing kinematic implications, thinking: {thinking1.content}; answer: {answer1.content}")
        all_thinking_1.append(thinking1)
        all_answer_1.append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + all_thinking_1 + all_answer_1, 
                                                    "Sub-task 1_1: Analyze kinematic implications." + 
                                                    " Given all the above thinking and answers, reason over them carefully and provide a final answer.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2_1: Integrate the extracted features and transformation analysis to determine how the total energy spectrum of the outgoing E particles changes in continuity, shape, and endpoint value." + reflect_inst
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking0, answer0, thinking1, answer1]
    subtask_desc_2 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking0, answer0, thinking1, answer1],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, integrating and refining energy spectrum changes, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], 
                                                "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", 
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining energy spectrum changes, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = "Sub-task 3_1: Evaluate the given answer choices against the integrated analysis to select the option that best describes the change in the energy spectrum of E particles." + \
                    " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    all_thinking_3 = []
    all_answer_3 = []
    subtask_desc_3 = {
        "subtask_id": "subtask_3_1",
        "instruction": debate_instr_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3):
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_3, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, evaluating answer choices, thinking: {thinking3.content}; answer: {answer3.content}")
        all_thinking_3.append(thinking3)
        all_answer_3.append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking_3 + all_answer_3, 
                                                    "Sub-task 3_1: Evaluate answer choices." + 
                                                    " Given all the above thinking and answers, reason over them carefully and provide a final answer.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
