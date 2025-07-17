async def forward_196(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    N_sc = self.max_sc
    cot_sc_instruction = "Sub-task 1: Extract and summarize all defining IR and 1H NMR spectral features (peak positions, multiplicities, integrations) from the given data."
    cot_sc_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_sc_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Given all the above thinking and answers, find the most consistent and correct summary of the spectral features."
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings + possible_answers, cot_sc_instruction + final_instr_1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2 = "Sub-task 2: Interpret the extracted features to classify functional groups and determine the substitution pattern on the benzene ring." + debate_instr
    N_round = self.max_round
    debate_agents_2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(N_round)]
    all_answer2 = [[] for _ in range(N_round)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":debate_instruction_2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"Debate"}
    for r in range(N_round):
        for agent in debate_agents_2:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking, answer = await agent(inputs, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking2[r].append(thinking)
            all_answer2[r].append(answer)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], debate_instruction_2 + final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction_3 = "Sub-task 3: Based on the functional classification and aromatic pattern, generate possible structural isomers from the provided choices and predict how red P/HI would modify the identified functional groups."
    cot_agent_3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction_3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction_4 = "Sub-task 4: Evaluate each candidate by matching its predicted IR/NMR features and HI/red P reaction outcome against the given data, then select the final product." + debate_instr
    debate_agents_4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(N_round)]
    all_answer4 = [[] for _ in range(N_round)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":debate_instruction_4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"Debate"}
    for r in range(N_round):
        for agent in debate_agents_4:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking, answer = await agent(inputs, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking4[r].append(thinking)
            all_answer4[r].append(answer)
    final_decision_agent_4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], debate_instruction_4 + final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs