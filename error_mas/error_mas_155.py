async def forward_155(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    sc_instruction_0 = "Sub-task 0: Determine the relative stereochemistry of the epoxide formed from (E)-oct-4-ene and (Z)-oct-4-ene upon mCPBA epoxidation."
    N0 = self.max_sc
    sc_agents_0 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N0)]
    possible_thinkings_0 = []
    possible_answers_0 = []
    subtask_desc0 = {"subtask_id":"subtask_0","instruction":sc_instruction_0,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N0):
        thinking0, answer0 = await sc_agents_0[i]([taskInfo], sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_0[i].id}, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings_0.append(thinking0)
        possible_answers_0.append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    final_instr_0 = "Sub-task 0: Synthesize and choose the most consistent and correct solution for the epoxide stereochemistry."
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, final_instr_0, is_sub_task=True)
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking":thinking0,"answer":answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])
    sc_instruction_1 = "Sub-task 1: Classify and count the stereoisomers formed: identify that the trans-epoxide (from E-alkene) is chiral (two enantiomers) and the cis-epoxide (from Z-alkene) is the meso form (one isomer)."
    N1 = self.max_sc
    sc_agents_1 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N1)]
    possible_thinkings_1 = []
    possible_answers_1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":sc_instruction_1,"context":["user query","thinking0","answer0"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1, answer1 = await sc_agents_1[i]([taskInfo, thinking0, answer0], sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_1[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings_1.append(thinking1)
        possible_answers_1.append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    final_instr_1 = "Sub-task 1: Synthesize and choose the most consistent and correct solution for stereoisomer classification and count."
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_1 + possible_answers_1, final_instr_1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction_2 = "Sub-task 2: Predict the number of peaks on a standard (achiral) reverse-phase HPLC column, knowing that enantiomers coelute but diastereomers separate."
    cot_agent_2 = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_instruction_2,"context":["user query","thinking1","answer1"],"agent_collaboration":"CoT","response":{"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction_3 = "Sub-task 3: Predict the number of peaks on a chiral HPLC column, knowing that enantiomers separate while the meso compound remains a single peak."
    cot_agent_3 = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction_3,"context":["user query","thinking1","answer1"],"agent_collaboration":"CoT","response":{"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs