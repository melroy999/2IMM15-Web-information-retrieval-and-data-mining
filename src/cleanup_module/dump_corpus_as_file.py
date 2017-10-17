import import_data.database as database
import cleanup_module.cleanup

# Create a cleanup instance so that we can use the special character removal there.
cp = cleanup_module.cleanup.get_cleanup_instance(database)

# We should not dump all of the files. We only dump the x longest ones.
papers = database.papers[:]
papers.sort(key= lambda x: len(x.paper_text), reverse=True)
with open("../../data/corpus.txt", "w") as output:
    for paper in papers[:500]:
        print(paper.id)
        output.writelines([cp.remove_control_characters(paper.paper_text)])
