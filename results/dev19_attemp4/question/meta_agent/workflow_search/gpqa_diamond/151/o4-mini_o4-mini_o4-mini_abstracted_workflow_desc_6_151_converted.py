async def forward_151(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0, Sub-task 1: CoT
    cot1_instruction = (
        "Sub-task 1: Extract and summarize all key experimental details from the query, "
        "including the quorum-sensing peptide treatment of yeast, shmoo formation, and planned ChIP–MS assay of active chromatin."
    )
    cot1_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot1_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot1_agent([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot1_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 0, Sub-task 2: Reflexion
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot2_instruction = (
        "Sub-task 2: List and describe the four candidate protein complexes from the query and their general chromatin roles. "
        + reflect_inst
    )
    cot2_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic2_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot2_instruction,
        "context": ["user query", "thinking1", "answer1"],
        "agent_collaboration": "Reflexion"
    }
    inputs2 = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot2_agent(inputs2, cot2_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot2_agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback2, correct2 = await critic2_agent([taskInfo, thinking2, answer2],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct' field.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic2_agent.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content.strip() == "True":
            break
        inputs2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot2_agent(inputs2, cot2_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot2_agent.id}, refined thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 0, Sub-task 3: Debate
    debate3_instruction = (
        "Sub-task 3: Classify each complex by its involvement in active transcription versus DNA replication or structural chromatin. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents3 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate3_instruction,
        "context": ["user query", "thinking2", "answer2"],
        "agent_collaboration": "Debate"
    }
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate3_instruction, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs3, debate3_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final3_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final3_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final3_agent(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Classify each complex..." + final3_instr,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final3_agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1, Sub-task 4: SC_CoT
    cot4_instruction = (
        "Sub-task 4: Assess the likelihood of detecting proteins from each complex in an active‐chromatin ChIP–MS experiment." 
    )
    N4 = self.max_sc
    cot4_agents = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N4)
    ]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot4_instruction,
        "context": ["user query", "thinking3", "answer3"],
        "agent_collaboration": "SC_CoT"
    }
    possible_thinkings4 = []
    possible_answers4 = []
    for i in range(N4):
        thinking4_i, answer4_i = await cot4_agents[i]([taskInfo, thinking3, answer3], cot4_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot4_agents[i].id}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final4_instr = "Given all the above thinking and answers, find the most consistent and correct assessment of detection likelihood."
    final4_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final4_agent(
        [taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4,
        "Sub-task 4: Select best assessment." + final4_instr,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final4_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2, Sub-task 5: SC_CoT
    cot5_instruction = (
        "Sub-task 5: Identify which complex will yield the fewest proteins in the active‐chromatin assay." 
    )
    N5 = self.max_sc
    cot5_agents = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N5)
    ]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot5_instruction,
        "context": ["user query", "thinking4", "answer4"],
        "agent_collaboration": "SC_CoT"
    }
    possible_thinkings5 = []
    possible_answers5 = []
    for i in range(N5):
        thinking5_i, answer5_i = await cot5_agents[i]([taskInfo, thinking4, answer4], cot5_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot5_agents[i].id}, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_thinkings5.append(thinking5_i)
        possible_answers5.append(answer5_i)
    final5_instr = "Given all the above thinking and answers, determine the one with the fewest proteins."
    final5_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final5_agent(
        [taskInfo, thinking4, answer4] + possible_thinkings5 + possible_answers5,
        "Sub-task 5: Choose the least observed complex." + final5_instr,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final5_agent.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs