async def forward_151(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC_CoT - Extract and summarize key query elements
    cot_sc_instruction = (
        "Sub-task 1: Extract and summarize the key elements of the query, including organisms, "
        "peptide treatment, observed phenotype, assay details, and candidate complexes."
    )
    N1 = self.max_sc
    cot_sc_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_sc_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent summary." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: CoT - Characterize each complex
    cot_instruction2 = (
        "Sub-task 2: Characterize the pre-initiation, pre-replication, enhancer protein, and nucleosome histone complexes "
        "in terms of composition, size, and association with active chromatin."
    )
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_instruction2,"context":["output of subtask_1"],"agent_collaboration":"CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Debate - Evaluate recoverability
    debate_instr3 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction3 = "Sub-task 3: Evaluate the expected abundance and ChIP–MS recoverability of each complex in actively transcribed shmoo chromatin." + debate_instr3
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N3 = self.max_round
    all_thinking3 = [[] for _ in range(N3)]
    all_answer3 = [[] for _ in range(N3)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction3,"context":["output of subtask_2"],"agent_collaboration":"Debate"}
    for r in range(N3):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                prev_inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(prev_inputs, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final evaluation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Reflexion - Select the complex with fewest proteins
    reflect_inst4 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction4 = "Sub-task 4: Compare the evaluated recoveries and select the complex yielding the fewest proteins." + reflect_inst4
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction4,"context":["responses of subtasks 1-3"],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4],
            "Please review and provide the limitations of the provided solution. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        inputs4 += [thinking4, answer4, feedback4]
        thinking4, answer4 = await cot_agent4(inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refined thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs